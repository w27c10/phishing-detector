"""
app.py — Flask inference server

Five-signal phishing detection pipeline:
  1. Rule-based URL structure scorer  (deterministic)
  2. DOM structure branch             (1-D CNN, neural)
  3. Metadata behaviour branch        (DNN, neural)
  4. Brand text impersonation         (keyword matching on visible page text)
  5. Domain age via WHOIS             (newly registered = high risk)
  6. Visual brand colour detection    (PIL analysis of page screenshot)

POST /analyze — stateless, no data written to disk.
"""

import base64
import io
import math
import os
import re
import sys
import threading
import time
import concurrent.futures
import json
import urllib.request
import urllib.parse
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlparse

try:
    import tldextract as _tldextract
    _HAS_TLDEXTRACT = True
except ImportError:
    _HAS_TLDEXTRACT = False

from bs4 import BeautifulSoup

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from feature_extractor import (
    extract_dom_features,
    extract_metadata_features,
    extract_url_features,
)

# ── App setup ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

# ── Load ONNX model once at startup ───────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'phishing_model.onnx')

try:
    _session      = ort.InferenceSession(MODEL_PATH)
    _input_names  = [inp.name for inp in _session.get_inputs()]
    _output_count = len(_session.get_outputs())
    print(f"[PhishingDetector] Model loaded. Inputs: {_input_names}")
except Exception as exc:
    print(f"[PhishingDetector] ERROR: Could not load model from {MODEL_PATH}\n  {exc}")
    sys.exit(1)

# Thread pool for non-blocking WHOIS lookups
_whois_pool  = concurrent.futures.ThreadPoolExecutor(max_workers=4)
_whois_cache: dict[str, float] = {}


# ── Google Safe Browsing API ───────────────────────────────────────────────────

_SB_KEY       = os.environ.get('SAFE_BROWSING_KEY', '')
_sb_cache:  dict[str, tuple[float, float]] = {}   # url → (score, timestamp)
_SB_TTL       = 1800   # cache results 30 min (Google's recommended minimum)

def _safe_browsing_score(url: str) -> float:
    """
    Queries Google Safe Browsing Lookup API v4.
    Returns 1.0 if the URL is flagged, 0.0 if clean or API unavailable.
    Results are cached for 30 minutes to stay within the free quota.
    """
    if not _SB_KEY or not url:
        return 0.0

    now = time.time()
    if url in _sb_cache:
        score, ts = _sb_cache[url]
        if now - ts < _SB_TTL:
            return score

    try:
        payload = json.dumps({
            'client': {'clientId': 'phishing-detector', 'clientVersion': '1.0'},
            'threatInfo': {
                'threatTypes':      ['SOCIAL_ENGINEERING', 'MALWARE', 'UNWANTED_SOFTWARE'],
                'platformTypes':    ['ANY_PLATFORM'],
                'threatEntryTypes': ['URL'],
                'threatEntries':    [{'url': url}],
            },
        }).encode()

        req = urllib.request.Request(
            f'https://safebrowsing.googleapis.com/v4/threatMatches:find?key={_SB_KEY}',
            data=payload,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            result = json.loads(resp.read())

        score = 1.0 if result.get('matches') else 0.0
    except Exception as exc:
        print(f'[PhishingDetector] Safe Browsing API error: {exc}')
        score = 0.0

    _sb_cache[url] = (score, now)
    return score


# ── Trusted domain system ──────────────────────────────────────────────────────
# Two-layer trust:
#   1. Manual whitelist — always trusted regardless of popularity.
#   2. Auto-updated Tranco Top-10 000 — refreshed daily, cached to disk.
#      Hosting platforms where attackers can create arbitrary subdomains
#      are excluded from auto-trust even if they rank highly.

_MANUAL_TRUSTED = {
    # Big tech — always safe regardless of Tranco cold-start
    'google.com', 'youtube.com', 'gmail.com', 'googleapis.com',
    'microsoft.com', 'live.com', 'outlook.com', 'office.com', 'bing.com',
    'apple.com', 'icloud.com',
    'amazon.com', 'amazonaws.com',
    'facebook.com', 'instagram.com', 'meta.com', 'whatsapp.com',
    'twitter.com', 'x.com',
    'linkedin.com',
    'netflix.com',
    'wikipedia.org', 'wikimedia.org',
    # Developer / deployment platforms
    'railway.com', 'github.com', 'gitlab.com', 'bitbucket.org',
    'vercel.com', 'netlify.com', 'heroku.com', 'render.com',
    'fly.io', 'digitalocean.com', 'cloudflare.com', 'npmjs.com',
    # Cloud consoles
    'aws.amazon.com', 'console.cloud.google.com', 'portal.azure.com',
    # Productivity / comms
    'notion.so', 'figma.com', 'canva.com', 'slack.com',
    'discord.com', 'zoom.us', 'reddit.com',
    # Payments
    'stripe.com', 'paypal.com',
}

# Platforms that allow arbitrary user subdomains — never auto-trust these
# even if they appear in the popularity list.
_HOSTING_PLATFORMS = {
    'github.io', 'gitlab.io', 'pages.dev', 'netlify.app', 'vercel.app',
    'wixsite.com', 'weebly.com', 'webflow.io', 'wordpress.com',
    'blogspot.com', 'webs.com', 'jimdo.com', 'squarespace.com',
    'glitch.me', 'replit.dev', 'repl.co', 'pythonanywhere.com',
    'mybluehost.me', 'biz.nf', 'site123.me', '000webhostapp.com',
}

_POPULAR_DOMAINS: set[str] = set()
_POPULAR_LOCK    = threading.Lock()
_POPULAR_CACHE   = os.path.join(os.path.dirname(__file__), '.popular_domains.txt')
_POPULAR_TOP_N   = 10_000


def _fetch_tranco() -> set[str]:
    """Download Tranco Top-N from their public permalink."""
    try:
        req = urllib.request.Request(
            'https://tranco-list.eu/download/latest/full',
            headers={'User-Agent': 'PhishingDetector/1.0 (research)'},
        )
        domains: set[str] = set()
        with urllib.request.urlopen(req, timeout=30) as resp:
            for i, line in enumerate(io.TextIOWrapper(resp)):
                if i >= _POPULAR_TOP_N:
                    break
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    d = parts[1].strip().lower()
                    if d and d not in _HOSTING_PLATFORMS:
                        domains.add(d)
        return domains
    except Exception as exc:
        print(f'[PhishingDetector] Tranco fetch failed: {exc}')
        return set()


def _load_popular_cache() -> set[str]:
    try:
        if os.path.exists(_POPULAR_CACHE):
            with open(_POPULAR_CACHE) as f:
                return {ln.strip() for ln in f if ln.strip()}
    except Exception:
        pass
    return set()


def _save_popular_cache(domains: set[str]) -> None:
    try:
        with open(_POPULAR_CACHE, 'w') as f:
            f.write('\n'.join(domains))
    except Exception:
        pass


def _popular_refresh_loop() -> None:
    global _POPULAR_DOMAINS
    while True:
        fresh = _fetch_tranco()
        if fresh:
            with _POPULAR_LOCK:
                _POPULAR_DOMAINS = fresh
            _save_popular_cache(fresh)
            print(f'[PhishingDetector] Popular domains refreshed: {len(fresh)} entries')
        time.sleep(86_400)   # re-fetch every 24 hours


# Seed from on-disk cache immediately (zero cold-start gap).
with _POPULAR_LOCK:
    _POPULAR_DOMAINS = _load_popular_cache()

# Kick off background refresh thread.
threading.Thread(target=_popular_refresh_loop, daemon=True).start()


def _is_trusted(url: str) -> bool:
    host = urlparse(url).hostname or ''

    # 1. Manual whitelist (exact host or any subdomain)
    if any(host == d or host.endswith('.' + d) for d in _MANUAL_TRUSTED):
        return True

    # 2. Auto-popular list: use the registered domain (last two labels)
    parts = host.split('.')
    reg   = '.'.join(parts[-2:]) if len(parts) >= 2 else host

    # Never auto-trust hosting platforms with user-controlled subdomains
    if reg in _HOSTING_PLATFORMS:
        return False

    with _POPULAR_LOCK:
        return reg in _POPULAR_DOMAINS


# ── Endpoint ───────────────────────────────────────────────────────────────────

@app.route('/analyze/url', methods=['POST'])
def analyze_url_only():
    """
    Fast URL-only analysis for the document_start phase.
    Runs only instant scorers (no WHOIS, no DOM).
    Hard-blocks only on Safe Browsing hits or gov impersonation (≥0.8).
    """
    body = request.get_json(force=True, silent=True) or {}
    url  = str(body.get('url', ''))
    if not url:
        return jsonify({'verdict': 'safe', 'threat_score': 0.0}), 400

    if _is_trusted(url):
        return jsonify({'verdict': 'safe', 'threat_score': 0.0})

    # Safe Browsing — cached, fast
    sb_score = _safe_browsing_score(url)
    if sb_score >= 1.0:
        return jsonify({'verdict': 'phishing', 'threat_score': 1.0, 'reason': 'safe_browsing'})

    url_score = _rule_url_score(url)
    gov_score = _gov_impersonation_score(url)

    # Hard-block only on high-confidence URL signals (no DOM to disambiguate)
    if gov_score >= 0.8:
        return jsonify({'verdict': 'phishing', 'threat_score': round(gov_score, 4), 'reason': 'gov_impersonation'})

    combined = max(url_score, gov_score)
    if combined >= 0.35:
        verdict = 'suspicious'
    else:
        verdict = 'safe'

    return jsonify({'verdict': verdict, 'threat_score': round(combined, 4)})


@app.route('/health', methods=['GET'])
def health():
    """Quick check that the deployed code has the expected brand list."""
    return jsonify({'brands': sorted(_BRAND_DOMAINS.keys()), 'usps_present': 'usps' in _BRAND_DOMAINS})


@app.route('/analyze', methods=['POST'])
def analyze():
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({'error': 'Invalid JSON body'}), 400

    url        = str(body.get('url',        ''))
    dom        = str(body.get('dom',        ''))
    text       = str(body.get('text',       ''))
    screenshot = str(body.get('screenshot', ''))

    if _is_trusted(url):
        return jsonify({'threat_score': 0.0, 'verdict': 'safe', 'explanation_details': {
            'url_diagnostic_message': 'Domain is in the trusted allowlist.'
        }})

    # ── Google Safe Browsing — check before running local models ──────────────
    sb_score = _safe_browsing_score(url)
    if sb_score >= 1.0:
        return jsonify({
            'threat_score': 1.0,
            'verdict':      'phishing',
            'explanation_details': {
                'safe_browsing_factor':  1.0,
                'safe_browsing_message': 'URL flagged by Google Safe Browsing.',
            },
        })

    # ── Neural branch inference ────────────────────────────────────────────────
    url_feat  = extract_url_features(url)
    dom_feat  = extract_dom_features(dom)
    meta_feat = extract_metadata_features(url, dom)

    inputs  = dict(zip(_input_names, [url_feat, dom_feat, meta_feat]))
    outputs = _session.run(None, inputs)

    dom_score  = float(outputs[2][0][0]) if _output_count > 2 else 0.0
    meta_score = float(outputs[3][0][0]) if _output_count > 3 else 0.0

    # ── Additional signal scorers ──────────────────────────────────────────────
    url_score       = _rule_url_score(url)
    brand_score     = _brand_text_score(url, text)
    gov_score       = _gov_impersonation_score(url)
    link_score      = _link_cluster_score(url, dom)
    # NLP two-layer partner check: suppress false positives for legitimate dealers
    if link_score >= 0.8:
        if _is_brand_partner(dom):                                          # Layer 1
            link_score = 0.0
        elif _llm_partner_check(dom, _get_dominant_domain(url, dom)):      # Layer 2
            link_score = 0.0
    dead_link_score = _dead_link_score(url, dom)
    age_score       = _domain_age_score(url)        # WHOIS, cached + timeout
    # ── Fusion ─────────────────────────────────────────────────────────────────
    # Hard overrides — structural signals only (gov keyword, link cluster, dead links).
    # brand_score is intentionally excluded: text keyword matches are too noisy
    # to hard-override alone (e.g. a legit page about Chrome will mention "Google").
    if gov_score >= 0.8 or link_score >= 0.8 or dead_link_score >= 0.8:
        final_score = max(gov_score, link_score, dead_link_score)
    else:
        final_score = (
            0.25 * url_score        +
            0.10 * age_score        +
            0.25 * dom_score        +
            0.15 * meta_score       +
            0.20 * brand_score      +
            0.05 * dead_link_score
        )

    # Adaptive threshold: suspicious URL lowers the bar for blocking.
    # When brand + metadata both fire at high confidence, tighten threshold:
    # these two signals together reliably indicate phishing even when DOM
    # score is slightly lower (e.g. SPA pages that render credentials lazily).
    _strong_signals = brand_score >= 0.8 and meta_score >= 0.8
    _base = 0.70 if _strong_signals else 0.75
    threshold = max(0.40, _base - url_score * 0.50)
    if final_score >= threshold:
        verdict = 'phishing'
    elif final_score >= 0.35:
        verdict = 'suspicious'
    else:
        verdict = 'safe'

    return jsonify({
        'threat_score': round(final_score, 4),
        'verdict':      verdict,
        'explanation_details': {
            'url_threat_factor':            round(url_score,    4),
            'url_diagnostic_message':       _url_message(url_score),
            'dom_threat_factor':            round(dom_score,    4),
            'dom_diagnostic_message':       _dom_message(dom_score),
            'metadata_threat_factor':       round(meta_score,   4),
            'metadata_diagnostic_message':  _meta_message(meta_score),
            'brand_threat_factor':          round(brand_score,  4),
            'brand_diagnostic_message':     _brand_message(brand_score),
            'gov_impersonation_factor':     round(gov_score,    4),
            'gov_impersonation_message':    _gov_message(gov_score),
            'link_cluster_factor':          round(link_score,        4),
            'link_cluster_message':         _link_cluster_message(link_score),
            'dead_link_factor':             round(dead_link_score,   4),
            'dead_link_message':            _dead_link_message(dead_link_score),
            'domain_age_factor':            round(age_score,    4),
            'domain_age_message':           _age_message(age_score),
        },
    })


# ── Domain parsing helper ──────────────────────────────────────────────────────

def _reg_domain(host: str) -> str:
    """
    Return the registered domain, correctly handling multi-part TLDs
    (.com.my, .co.uk, .com.sg, etc.) via tldextract when available.
    Falls back to naive last-two-label splitting if tldextract is absent.
    """
    if _HAS_TLDEXTRACT:
        try:
            r = _tldextract.extract(host)
            v = r.top_domain_under_public_suffix
            if v:
                return v
        except Exception:
            pass
    parts = host.split('.')
    return '.'.join(parts[-2:]) if len(parts) >= 2 else host


# ── 1. Rule-based URL risk scorer ──────────────────────────────────────────────

def _rule_url_score(url: str) -> float:
    """Deterministic URL risk score [0, 1] based on structural patterns."""
    try:
        parsed = urlparse(url)
        host   = parsed.hostname or ''
    except Exception:
        return 0.0

    reg_domain = _reg_domain(host)
    if _HAS_TLDEXTRACT:
        try:
            _ext   = _tldextract.extract(host)
            tld    = _ext.suffix.split('.')[-1] if _ext.suffix else (host.split('.')[-1])
            subdomains = [s for s in _ext.subdomain.split('.') if s]
        except Exception:
            parts = host.split('.'); tld = parts[-1]; subdomains = parts[:-2]
    else:
        parts = host.split('.')
        tld   = parts[-1] if parts else ''
        subdomains = parts[:-2]

    risk = 0.0

    # Subdomain depth
    n_sub = len(subdomains)
    if n_sub >= 3:
        risk += 0.50
    elif n_sub == 2:
        risk += 0.30

    # Subdomain entropy (random-looking labels)
    if subdomains:
        sub_str = ''.join(subdomains)
        n = len(sub_str)
        if n > 0:
            freq: dict[str, int] = {}
            for c in sub_str:
                freq[c] = freq.get(c, 0) + 1
            entropy = -sum((f / n) * math.log2(f / n) for f in freq.values())
            if entropy > 3.5:
                risk += 0.25
            elif entropy > 2.5:
                risk += 0.10

    # IP address as host
    if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host):
        risk += 0.60

    # @ symbol
    if '@' in url:
        risk += 0.50

    # Risky TLDs
    RISKY_TLDS = {'tk','ml','ga','cf','gq','xyz','top','club','work','click',
                  'link','online','site','website','info','biz','pw','cc','su','ru',
                  'shop','store','app','tech','live','vip','pro','cfd','sbs','cyou'}
    if tld.lower() in RISKY_TLDS:
        risk += 0.20

    # Brand keyword in subdomain but not in registered domain
    BRANDS = {
        # Finance / Payment
        'paypal','visa','mastercard','amex','westernunion','wise','revolut',
        'venmo','cashapp','alipay','klarna','afterpay','affirm','capitalone',
        'discover','monzo','n26','chime','nubank',
        # Banking
        'bankofamerica','chase','wellsfargo','hsbc','citibank',
        'barclays','santander','lloyds','natwest','dbs','maybank','ocbc','uob',
        'commbank','anz','westpac','rbc','scotiabank','bmo','ing','rabobank',
        'hdfc','icici','sbi',
        # Big Tech
        'apple','amazon','google','microsoft','netflix','facebook','instagram',
        'whatsapp','dropbox','adobe','salesforce','atlassian','oracle',
        'shopify','twilio','hubspot',
        # Logistics
        'dhl','fedex','ups','usps','singpost','royalmail',
        # Crypto
        'bybit','binance','coinbase','okx','kraken','kucoin','huobi','htx',
        'bitfinex','gemini','mexc','bitget','bitmex','phemex','etoro',
        'robinhood','metamask','uniswap','opensea','ledger','trezor',
        # E-commerce
        'ebay','aliexpress','shopee','lazada','walmart','etsy','rakuten',
        'shein','temu','zalando','asos','bestbuy','costco','ikea','nike','adidas',
        # Social / Communication
        'tiktok','snapchat','telegram','pinterest','wechat','twitch',
        # Gaming
        'roblox','epicgames','playstation','xbox','blizzard','nintendo',
        'ubisoft','activision','rockstar','minecraft',
        # Streaming
        'spotify','disney','hulu','paramount',
        # Travel
        'airbnb','marriott','hilton','emirates','expedia','uber','grab',
        # Telecom
        'verizon','xfinity','comcast','vodafone','singtel','maxis',
        # Insurance / Healthcare
        'axa','allianz','prudential','metlife','cigna','cvs','walgreens',
    }
    sub_text = ' '.join(subdomains).lower()
    reg_text = reg_domain.lower()
    for brand in BRANDS:
        pattern = r'\b' + re.escape(brand) + r'\b'
        if re.search(pattern, sub_text) and not re.search(pattern, reg_text):
            risk += 0.50
            break

    # Path TLD spoofing — e.g. /com or /net pretending to be a domain extension
    FAKE_EXTS = {'com', 'net', 'org', 'gov', 'uk', 'us', 'edu'}
    path_parts = [p for p in (parsed.path or '').strip('/').split('/') if p]
    if path_parts and path_parts[0].lower() in FAKE_EXTS:
        risk += 0.20

    # URL length
    if len(url) > 150:
        risk += 0.10
    if len(url) > 250:
        risk += 0.10

    # Non-HTTPS
    if parsed.scheme != 'https':
        risk += 0.10

    return min(risk, 1.0)


# ── 2. Government keyword impersonation scorer ────────────────────────────────

# Genuine government second-level domains to exclude from the check.
_REAL_GOV_SLDS = {'gov.uk', 'gov.au', 'gov.nz', 'govt.nz', 'gov.sg', 'gov.in',
                  'gov.za', 'gov.ca', 'gov.ie', 'gov.br', 'gob.mx', 'gouv.fr'}

def _gov_impersonation_score(url: str) -> float:
    """
    Returns 1.0 if 'gov' appears as a standalone hyphen-delimited word in the
    hostname but the domain is not a genuine government TLD or SLD.
    """
    try:
        parsed = urlparse(url)
        host   = parsed.hostname or ''
        parts  = host.split('.')
        tld    = parts[-1].lower()

        if tld in ('gov', 'mil'):
            return 0.0
        if len(parts) >= 2 and f'{parts[-2].lower()}.{tld}' in _REAL_GOV_SLDS:
            return 0.0

        for part in parts:
            if 'gov' in part.lower().split('-'):
                return 1.0
        return 0.0
    except Exception:
        return 0.0


# ── 3. Link cluster impersonation scorer ──────────────────────────────────────

# Contact/social platforms that legitimate sites commonly link to en-masse
# (e.g. multiple WhatsApp CTA buttons, social media icons in footer).
# These should never be treated as "dominant external domain" evidence of cloning.
_CONTACT_PLATFORMS = {
    'wa.me', 'whatsapp.com',                        # WhatsApp
    't.me', 'telegram.org', 'telegram.me',          # Telegram
    'line.me',                                       # Line
    'facebook.com', 'fb.com', 'fb.me',              # Facebook
    'instagram.com',                                 # Instagram
    'twitter.com', 'x.com',                         # Twitter / X
    'linkedin.com',                                  # LinkedIn
    'youtube.com', 'youtu.be',                      # YouTube
    'tiktok.com',                                    # TikTok
    'pinterest.com',                                 # Pinterest
    'maps.google.com', 'goo.gl', 'maps.app.goo.gl', # Google Maps
    'wordpress.org', 'wordpress.com',               # WordPress CMS credits
}

def _link_cluster_score(url: str, html: str) -> float:
    """
    Returns a high score when the overwhelming majority of external links on
    a page point to a single domain that differs from the hosting domain.
    This is the hallmark of a cloned/impersonation page: all nav links point
    back to the real site while the page itself sits on an attacker's domain.
    """
    if not html:
        return 0.0
    try:
        soup = BeautifulSoup(html, 'html.parser')
        host = urlparse(url).hostname or ''

        page_reg = _reg_domain(host)
        external_domains: list[str] = []
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
            netloc = urlparse(href).netloc
            if not netloc:
                continue
            reg = _reg_domain(netloc)
            if not reg or reg == page_reg:
                continue                              # internal / same-site link
            # Skip known contact/social platforms — legitimate sites link to
            # these en-masse (multiple WhatsApp buttons, social icons, etc.)
            if netloc.lower() in _CONTACT_PLATFORMS or reg.lower() in _CONTACT_PLATFORMS:
                continue
            external_domains.append(reg.lower())

        if len(external_domains) < 5:
            return 0.0

        top_domain, top_count = Counter(external_domains).most_common(1)[0]
        ratio = top_count / len(external_domains)

        # >80 % of all external links pointing to one domain = strong clone signal
        return round(ratio, 4) if ratio >= 0.80 else 0.0
    except Exception:
        return 0.0


# ── 4. NLP brand-partner disambiguation ───────────────────────────────────────
#
# When link_cluster_score fires (≥ 0.8), it could be:
#   A) A phishing clone that links back to the real site (bad)
#   B) A legitimate dealer/partner whose site links to the official brand (good)
#
# Two-layer check:
#   Layer 1 — fast regex: look for "authorized dealer / official partner" keywords
#   Layer 2 — Gemini API: semantic classification for multilingual / keyword-absent cases

GEMINI_KEY = os.environ.get('GEMINI_API_KEY', '')

_PARTNER_PATTERNS = [
    r'authorized\s+(dealer|distributor|reseller|partner|agent|retailer|service)',
    r'authorised\s+(dealer|distributor|reseller|partner|agent|retailer|service)',
    r'official\s+(dealer|distributor|reseller|partner|agent|retailer)',
    r'certified\s+(dealer|distributor|partner|agent)',
    r'appointed\s+(dealer|distributor|partner|agent)',
    r'\bdealership\b',
    r'official\s+representative',
    r'authorized\s+service\s+cent(er|re)',
    r'authorised\s+service\s+cent(er|re)',
    r'premium\s+(reseller|partner)',
    r'reseller\s+of',
    r'distributor\s+of',
]


def _is_brand_partner(html: str) -> bool:
    """Layer 1: scan full page text for brand partner / dealer keyword patterns."""
    if not html:
        return False
    try:
        text = BeautifulSoup(html, 'html.parser').get_text(separator=' ').lower()
        return any(re.search(p, text) for p in _PARTNER_PATTERNS)
    except Exception:
        return False


def _get_dominant_domain(url: str, html: str) -> str:
    """Return the most-linked external registered domain (used as LLM context)."""
    if not html:
        return ''
    try:
        soup = BeautifulSoup(html, 'html.parser')
        host = urlparse(url).hostname or ''
        external: list[str] = []
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
            netloc = urlparse(href).netloc
            if not netloc or host in netloc:
                continue
            parts = netloc.split('.')
            external.append(('.'.join(parts[-2:]) if len(parts) >= 2 else netloc).lower())
        return Counter(external).most_common(1)[0][0] if external else ''
    except Exception:
        return ''


def _llm_partner_check(html: str, dominant_domain: str) -> bool:
    """
    Layer 2: ask Gemini 1.5 Flash to classify the page as IMPERSONATION or PARTNER.
    Only called when Layer 1 finds no keyword evidence.
    Returns True (suppress phishing flag) if Gemini says PARTNER.
    """
    if not GEMINI_KEY or not html or not dominant_domain:
        return False
    try:
        page_text = BeautifulSoup(html, 'html.parser').get_text(separator=' ')
        page_text = ' '.join(page_text.split())[:600]   # clean whitespace, cap length

        prompt = (
            f'Page text: "{page_text}"\n'
            f'Most-linked external domain: {dominant_domain}\n\n'
            f'Is this page claiming to BE {dominant_domain} (phishing/impersonation), '
            f'or is it an AUTHORIZED DEALER / PARTNER / RESELLER of {dominant_domain} '
            f'(legitimate business)?\n'
            f'Reply with exactly one word: IMPERSONATION or PARTNER.'
        )
        payload = json.dumps({
            'contents': [{'parts': [{'text': prompt}]}]
        }).encode()
        req = urllib.request.Request(
            f'https://generativelanguage.googleapis.com/v1beta/models/'
            f'gemini-1.5-flash:generateContent?key={GEMINI_KEY}',
            data=payload,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
        answer = result['candidates'][0]['content']['parts'][0]['text'].strip().upper()
        print(f'[PhishingDetector] Gemini partner check → {answer} ({dominant_domain})')
        return 'PARTNER' in answer
    except Exception as exc:
        print(f'[PhishingDetector] Gemini partner check failed: {exc}')
        return False


# ── 5. Dead / decorative link scorer ──────────────────────────────────────────

def _dead_link_score(url: str, html: str) -> float:
    """
    Measures the ratio of decorative (non-functional) anchor tags.

    Legitimate sites wire up every navigation link. Phishing pages are
    hastily cloned HTML — the data-collection form works but most
    navigation buttons are left as href="#" / javascript:void(0) shells
    with no onclick or router data attribute.

    Only counts links with no real href AND no JS handler AND no data-*
    routing attributes, so modern SPA router links (href="/path") are
    not penalised.
    """
    if not html:
        return 0.0
    try:
        soup  = BeautifulSoup(html, 'html.parser')
        links = soup.find_all('a')
        if len(links) < 5:
            return 0.0

        _DEAD_HREFS = {'', '#', 'javascript:', 'javascript:void(0)',
                       'javascript:void(0);', 'javascript: void(0)'}

        dead = 0
        for a in links:
            href        = a.get('href', '').strip().lower()
            has_real    = href and href not in _DEAD_HREFS and not href.startswith('javascript:')
            has_onclick  = bool(a.get('onclick'))
            # Only treat routing-specific data attributes as "functional"
            _ROUTING_DATA = {'data-href', 'data-to', 'data-url', 'data-route',
                             'data-link', 'data-path', 'data-navigate'}
            has_data = any(k in _ROUTING_DATA for k in a.attrs)
            if not has_real and not has_onclick and not has_data:
                dead += 1

        ratio = dead / len(links)
        if ratio < 0.50:
            return 0.0

        # On a free hosting platform, dead links are a much stronger phishing signal:
        # attacker cloned a page but only wired up the payment/credential form.
        host  = urlparse(url).hostname or ''
        parts = host.split('.')
        reg   = '.'.join(parts[-2:]) if len(parts) >= 2 else host
        if reg in _HOSTING_PLATFORMS:
            return min(ratio + 0.30, 1.0)   # 50 % dead → 0.80, 100 % dead → 1.0

        return round(ratio, 4)
    except Exception:
        return 0.0


# ── 5. Brand text impersonation scorer ─────────────────────────────────────────

# Maps brand keywords to their legitimate domain suffixes.
_BRAND_DOMAINS: dict[str, list[str]] = {
    'paypal':          ['paypal.com'],
    'apple':           ['apple.com', 'icloud.com'],
    'google':          ['google.com', 'gmail.com', 'accounts.google'],
    'microsoft':       ['microsoft.com', 'live.com', 'outlook.com', 'office.com'],
    'facebook':        ['facebook.com', 'meta.com'],
    'instagram':       ['instagram.com'],
    'amazon':          ['amazon.com', 'amazonaws.com'],
    'netflix':         ['netflix.com'],
    'twitter':         ['twitter.com', 'x.com'],
    'linkedin':        ['linkedin.com'],
    'dropbox':         ['dropbox.com'],
    'chase':           ['chase.com', 'jpmorganchase.com'],
    'bank of america': ['bankofamerica.com'],
    'wells fargo':     ['wellsfargo.com'],
    'citibank':        ['citibank.com', 'citi.com'],
    'hsbc':            ['hsbc.com'],
    'dhl':             ['dhl.com'],
    'fedex':           ['fedex.com'],
    'ups':             ['ups.com'],
    'usps':            ['usps.com', 'usps.gov'],
    'steam':           ['steampowered.com', 'steamcommunity.com'],
    # ── Crypto exchanges & wallets ─────────────────────────────────────────────
    'binance':         ['binance.com'],
    'coinbase':        ['coinbase.com'],
    'crypto':          ['crypto.com'],
    'bybit':           ['bybit.com'],
    'okx':             ['okx.com'],
    'kraken':          ['kraken.com'],
    'kucoin':          ['kucoin.com'],
    'huobi':           ['huobi.com'],
    'htx':             ['htx.com'],
    'bitfinex':        ['bitfinex.com'],
    'gemini':          ['gemini.com'],
    'mexc':            ['mexc.com'],
    'bitget':          ['bitget.com'],
    'bitmex':          ['bitmex.com'],
    'phemex':              ['phemex.com'],
    'etoro':               ['etoro.com'],
    'robinhood':           ['robinhood.com'],
    'metamask':            ['metamask.io'],
    'trust wallet':        ['trustwallet.com'],
    'ledger':              ['ledger.com'],
    'trezor':              ['trezor.io'],
    'uniswap':             ['uniswap.org'],
    'opensea':             ['opensea.io'],
    'blockchain.com':      ['blockchain.com'],
    # ── Financial / Payment ────────────────────────────────────────────────────
    'visa':                ['visa.com'],
    'mastercard':          ['mastercard.com'],
    'american express':    ['americanexpress.com'],
    'amex':                ['americanexpress.com'],
    'western union':       ['westernunion.com'],
    'wise':                ['wise.com'],
    'revolut':             ['revolut.com'],
    'venmo':               ['venmo.com'],
    'cash app':            ['cash.app', 'cashapp.com'],
    'alipay':              ['alipay.com'],
    'klarna':              ['klarna.com'],
    'afterpay':            ['afterpay.com'],
    'affirm':              ['affirm.com'],
    'capital one':         ['capitalone.com'],
    'discover':            ['discover.com'],
    'monzo':               ['monzo.com'],
    'n26':                 ['n26.com'],
    'chime':               ['chime.com'],
    'nubank':              ['nubank.com'],
    # ── Banking ────────────────────────────────────────────────────────────────
    'barclays':            ['barclays.com'],
    'santander':           ['santander.com'],
    'lloyds':              ['lloyds.com'],
    'natwest':             ['natwest.com'],
    'standard chartered':  ['sc.com'],
    'dbs':                 ['dbs.com'],
    'maybank':             ['maybank.com'],
    'ocbc':                ['ocbc.com'],
    'uob':                 ['uob.com'],
    'commbank':            ['commbank.com.au'],
    'commonwealth bank':   ['commbank.com.au'],
    'anz':                 ['anz.com'],
    'westpac':             ['westpac.com.au'],
    'rbc':                 ['rbc.com'],
    'scotiabank':          ['scotiabank.com'],
    'bmo':                 ['bmo.com'],
    'ing':                 ['ing.com'],
    'rabobank':            ['rabobank.com'],
    'hdfc':                ['hdfcbank.com'],
    'icici':               ['icicibank.com'],
    'sbi':                 ['sbi.co.in'],
    # ── E-commerce / Retail ────────────────────────────────────────────────────
    'ebay':                ['ebay.com'],
    'aliexpress':          ['aliexpress.com'],
    'shopee':              ['shopee.com'],
    'lazada':              ['lazada.com'],
    'walmart':             ['walmart.com'],
    'etsy':                ['etsy.com'],
    'rakuten':             ['rakuten.com'],
    'shein':               ['shein.com'],
    'temu':                ['temu.com'],
    'zalando':             ['zalando.com'],
    'asos':                ['asos.com'],
    'bestbuy':             ['bestbuy.com'],
    'best buy':            ['bestbuy.com'],
    'costco':              ['costco.com'],
    'ikea':                ['ikea.com'],
    'nike':                ['nike.com'],
    'adidas':              ['adidas.com'],
    # ── Social / Communication ─────────────────────────────────────────────────
    'tiktok':              ['tiktok.com'],
    'snapchat':            ['snapchat.com'],
    'telegram':            ['telegram.org'],
    'pinterest':           ['pinterest.com'],
    'wechat':              ['wechat.com'],
    'line':                ['line.me'],
    'signal':              ['signal.org'],
    'twitch':              ['twitch.tv'],
    # ── Gaming ─────────────────────────────────────────────────────────────────
    'roblox':              ['roblox.com'],
    'epic games':          ['epicgames.com'],
    'playstation':         ['playstation.com'],
    'xbox':                ['xbox.com'],
    'riot games':          ['riotgames.com'],
    'blizzard':            ['blizzard.com'],
    'nintendo':            ['nintendo.com'],
    'electronic arts':     ['ea.com'],
    'ubisoft':             ['ubisoft.com'],
    'activision':          ['activision.com'],
    'rockstar':            ['rockstargames.com'],
    'minecraft':           ['minecraft.net'],
    # ── Streaming / Entertainment ──────────────────────────────────────────────
    'spotify':             ['spotify.com'],
    'disney':              ['disney.com', 'disneyplus.com'],
    'hbo':                 ['hbo.com', 'max.com'],
    'hulu':                ['hulu.com'],
    'paramount':           ['paramountplus.com'],
    # ── Travel / Hospitality ───────────────────────────────────────────────────
    'booking':             ['booking.com'],
    'expedia':             ['expedia.com'],
    'airbnb':              ['airbnb.com'],
    'marriott':            ['marriott.com'],
    'hilton':              ['hilton.com'],
    'emirates':            ['emirates.com'],
    'singapore airlines':  ['singaporeair.com'],
    'british airways':     ['britishairways.com'],
    'uber':                ['uber.com'],
    'grab':                ['grab.com'],
    # ── Logistics ─────────────────────────────────────────────────────────────
    'royal mail':          ['royalmail.com'],
    'australia post':      ['auspost.com.au'],
    'japan post':          ['japanpost.jp'],
    'pos malaysia':        ['pos.com.my'],
    'singpost':            ['singpost.com'],
    # ── Enterprise Software ────────────────────────────────────────────────────
    'adobe':               ['adobe.com'],
    'salesforce':          ['salesforce.com'],
    'atlassian':           ['atlassian.com'],
    'oracle':              ['oracle.com'],
    'shopify':             ['shopify.com'],
    'twilio':              ['twilio.com'],
    'mailchimp':           ['mailchimp.com'],
    'hubspot':             ['hubspot.com'],
    # ── Telecom ────────────────────────────────────────────────────────────────
    'at&t':                ['att.com'],
    'verizon':             ['verizon.com'],
    't-mobile':            ['t-mobile.com'],
    'xfinity':             ['xfinity.com'],
    'comcast':             ['comcast.com'],
    'vodafone':            ['vodafone.com'],
    'singtel':             ['singtel.com'],
    'maxis':               ['maxis.com.my'],
    'globe':               ['globe.com.ph'],
    # ── Insurance / Healthcare ─────────────────────────────────────────────────
    'axa':                 ['axa.com'],
    'allianz':             ['allianz.com'],
    'prudential':          ['prudential.com'],
    'aia':                 ['aia.com'],
    'metlife':             ['metlife.com'],
    'cigna':               ['cigna.com'],
    'cvs':                 ['cvs.com'],
    'walgreens':           ['walgreens.com'],
    # ── Government / Public Services ──────────────────────────────────────────
    'irs':                 ['irs.gov'],
    'hmrc':                ['hmrc.gov.uk', 'gov.uk'],
    'nhs':                 ['nhs.uk'],
    'medicare':            ['medicare.gov'],
    'centrelink':          ['servicesaustralia.gov.au'],
}

# Brands that are short (≤3 chars) or common English words.
# These require ≥2 occurrences in page text to fire, reducing false positives
# where the word appears incidentally (e.g. "ing" in "savings", "line" in "online").
_AMBIGUOUS_BRANDS: frozenset = frozenset({
    # ≤3 chars
    'ing', 'dbs', 'uob', 'aia', 'anz', 'sbi', 'rbc', 'bmo',
    'htx', 'okx', 'cvs', 'nhs', 'hbo', 'ups', 'axa',
    # Common English words
    'grab', 'line', 'wise', 'signal', 'globe', 'chase',
    'discover', 'booking', 'paramount', 'affirm', 'ledger',
    'crypto', 'steam', 'chime',
})

def _brand_text_score(url: str, text: str) -> float:
    """
    Returns 1.0 if a recognised brand name appears in the page's visible text
    but the URL does not belong to that brand's legitimate domain.

    Uses word-boundary matching (\b) so 'ing' won't match 'savings', etc.
    Ambiguous/short brands require ≥2 occurrences to fire.
    """
    host       = urlparse(url).hostname or ''
    text_lower = text.lower()
    reg_domain = _reg_domain(host).lower()

    for brand, legit_domains in _BRAND_DOMAINS.items():
        pattern   = r'\b' + re.escape(brand) + r'\b'
        min_count = 2 if brand in _AMBIGUOUS_BRANDS else 1
        if len(re.findall(pattern, text_lower)) >= min_count:
            if not any(d in host for d in legit_domains):
                # Skip if the brand name IS the registered domain
                # e.g. 'google' in 'google.dev' → legitimate Google property
                if re.search(pattern, reg_domain):
                    continue
                return 1.0
    return 0.0


# ── 3. Domain age scorer (WHOIS) ───────────────────────────────────────────────

def _whois_lookup(domain: str) -> float:
    """Run in a thread. Returns age risk score [0, 1]."""
    try:
        import whois as whois_lib
        w        = whois_lib.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if not creation:
            return 0.0
        if getattr(creation, 'tzinfo', None):
            age_days = (datetime.now(timezone.utc) - creation).days
        else:
            age_days = (datetime.now() - creation).days
        # 0 days → 1.0 risk; 365+ days → 0.0 risk
        return max(0.0, 1.0 - age_days / 365.0)
    except Exception:
        return 0.0


def _domain_age_score(url: str) -> float:
    """WHOIS domain age with 3-second timeout and in-memory cache."""
    host   = urlparse(url).hostname or ''
    domain = re.sub(r'^www\.', '', host)
    if not domain:
        return 0.0
    if domain in _whois_cache:
        return _whois_cache[domain]
    try:
        score = _whois_pool.submit(_whois_lookup, domain).result(timeout=3.0)
    except concurrent.futures.TimeoutError:
        score = 0.0
    _whois_cache[domain] = score
    return score


# ── 4. Visual brand colour scorer ──────────────────────────────────────────────

# Primary brand colours (R, G, B). Each brand has 1-4 characteristic colours.
_BRAND_COLOURS: dict[str, list[tuple[int, int, int]]] = {
    'paypal':    [(0, 48, 135),   (0, 156, 222)],
    'facebook':  [(24, 119, 242), (66, 103, 178)],
    'google':    [(66, 133, 244), (234, 67, 53), (251, 188, 4), (52, 168, 83)],
    'microsoft': [(0, 120, 212),  (255, 67, 0),  (127, 186, 0), (255, 185, 0)],
    'twitter':   [(29, 161, 242)],
    'amazon':    [(255, 153, 0),  (35, 47, 62)],
    'netflix':   [(229, 9, 20)],
    'linkedin':  [(0, 119, 181)],
}

_BRAND_COLOUR_DOMAINS: dict[str, list[str]] = {
    'paypal':    ['paypal.com'],
    'facebook':  ['facebook.com', 'meta.com'],
    'google':    ['google.com', 'gmail.com'],
    'microsoft': ['microsoft.com', 'live.com', 'outlook.com', 'office.com'],
    'twitter':   ['twitter.com', 'x.com'],
    'amazon':    ['amazon.com'],
    'netflix':   ['netflix.com'],
    'linkedin':  ['linkedin.com'],
}

def _colour_dist(a: tuple, b: tuple) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def _visual_score(url: str, screenshot_b64: str) -> float:
    """
    Decode the JPEG screenshot, downsample to 100×75, count pixels that match
    known brand colour palettes. If a brand's colours dominate the viewport
    but the URL doesn't belong to that brand, return a high risk score.
    """
    if not screenshot_b64:
        return 0.0
    try:
        img_data = base64.b64decode(screenshot_b64)
        img      = Image.open(io.BytesIO(img_data)).convert('RGB')
        img      = img.resize((100, 75), Image.LANCZOS)
        pixels   = list(img.getdata())
        n        = len(pixels)
        host     = urlparse(url).hostname or ''

        for brand, colours in _BRAND_COLOURS.items():
            matched = sum(
                1 for p in pixels
                if any(_colour_dist(p, c) < 25 for c in colours)
            )
            coverage = matched / n
            if coverage > 0.28:   # brand colour covers >28 % of the viewport
                legit = _BRAND_COLOUR_DOMAINS.get(brand, [])
                if not any(d in host for d in legit):
                    return min(coverage * 3.0, 1.0)
        return 0.0
    except Exception:
        return 0.0


# ── Diagnostic message generators ──────────────────────────────────────────────

def _url_message(score: float) -> str:
    if score >= 0.75:
        return 'Anomalous brand keyword or structural pattern detected in URL.'
    if score >= 0.40:
        return 'Suspicious subdomain structure or URL composition detected.'
    return 'URL structure appears within normal parameters.'

def _dom_message(score: float) -> str:
    if score >= 0.75:
        return 'High structural correlation with known phishing page templates.'
    if score >= 0.50:
        return 'Page layout contains elements associated with credential harvesting.'
    return 'Page DOM structure appears within normal parameters.'

def _meta_message(score: float) -> str:
    if score >= 0.75:
        return 'Suspicious data routing: form action targets an unassociated domain.'
    if score >= 0.50:
        return 'Elevated ratio of external resource references detected.'
    return 'Page metadata and behavioural signals appear within normal parameters.'

def _brand_message(score: float) -> str:
    if score >= 0.8:
        return 'Brand name detected in page content — domain does not belong to that brand.'
    return 'No brand impersonation detected in visible page text.'

def _dead_link_message(score: float) -> str:
    if score >= 0.5:
        return f'~{round(score * 100)}% of links are non-functional — page may be a cloned template.'
    return 'Link functionality appears normal.'

def _link_cluster_message(score: float) -> str:
    if score >= 0.8:
        return 'External links cluster to one domain — page is likely a clone of that site.'
    return 'External links distributed normally.'

def _gov_message(score: float) -> str:
    if score >= 0.8:
        return 'Government keyword in non-.gov domain — high confidence impersonation.'
    return 'No government keyword impersonation detected.'

def _age_message(score: float) -> str:
    if score >= 0.90:
        return 'Domain registered within the last 14 days — extremely high risk.'
    if score >= 0.70:
        return 'Domain registered within the last 3 months — elevated risk.'
    if score >= 0.40:
        return 'Domain registered within the last 6 months — moderate risk.'
    return 'Domain has sufficient registration history.'

def _visual_message(score: float) -> str:
    if score >= 0.7:
        return 'Brand colour signature detected in page screenshot — domain mismatch confirmed.'
    if score >= 0.3:
        return 'Partial brand colour match detected in page screenshot.'
    return 'No brand colour signature detected in visual analysis.'


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
