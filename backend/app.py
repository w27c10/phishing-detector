"""
app.py — Flask inference server

Five-signal phishing detection pipeline:
  1. Rule-based URL structure scorer  (deterministic)
  2. DOM structure branch             (1-D CNN, neural)
  3. Metadata behaviour branch        (DNN, neural)
  4. Brand text impersonation         (keyword matching on visible page text)
  5. Domain age via WHOIS             (newly registered = high risk)


POST /analyze — stateless, no data written to disk.
"""

import concurrent.futures
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

# Separate pool for HTTP link-reachability checks (avoid starving WHOIS workers)
_link_check_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)


# ── Google Safe Browsing API ───────────────────────────────────────────────────

_SB_KEY       = os.environ.get('SAFE_BROWSING_KEY', '')
_sb_cache:  dict[str, tuple[float, float]] = {}   # url → (score, timestamp)
_SB_TTL       = 1800   # cache results 30 min (Google's recommended minimum)

_gemini_brand_cache: dict[str, tuple[float, float]] = {}  # url → (score, timestamp)
_GEMINI_BRAND_TTL = 1800

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


def _fetch_tranco_domains() -> set[str]:
    """
    Fetch Tranco top domains via two-step:
    1. GET /latest_list to parse the current list ID.
    2. Stream /download/{ID}/full (raw CSV, rank,domain) up to _POPULAR_TOP_N rows.
    """
    try:
        id_req = urllib.request.Request(
            'https://tranco-list.eu/latest_list',
            headers={'User-Agent': 'PhishingDetector/1.0 (research)'},
        )
        with urllib.request.urlopen(id_req, timeout=10) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
        m = re.search(r'list with ID\s+([A-Z0-9]+)', html, re.IGNORECASE)
        if not m:
            print('[PhishingDetector] Tranco: could not parse list ID', flush=True)
            return set()
        list_id = m.group(1)
        csv_url = f'https://tranco-list.eu/download/{list_id}/full'
        csv_req = urllib.request.Request(
            csv_url, headers={'User-Agent': 'PhishingDetector/1.0 (research)'},
        )
        domains: set[str] = set()
        with urllib.request.urlopen(csv_req, timeout=30) as resp:
            for line in io.TextIOWrapper(resp):
                if len(domains) >= _POPULAR_TOP_N:
                    break
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    d = parts[1].strip().lower()
                    if d and d not in _HOSTING_PLATFORMS:
                        domains.add(d)
        return domains
    except Exception as exc:
        print(f'[PhishingDetector] Tranco fetch failed: {exc}', flush=True)
        return set()


def _fetch_popular_domains() -> set[str]:
    """Download top domains from Majestic Million (primary) or Tranco (fallback)."""
    # Majestic Million — stable direct CSV, format: GlobalRank,TldRank,Domain,...
    try:
        req = urllib.request.Request(
            'https://downloads.majestic.com/majestic_million.csv',
            headers={'User-Agent': 'PhishingDetector/1.0 (research)'},
        )
        domains: set[str] = set()
        with urllib.request.urlopen(req, timeout=30) as resp:
            for i, line in enumerate(io.TextIOWrapper(resp)):
                if i == 0:
                    continue          # skip CSV header row
                if len(domains) >= _POPULAR_TOP_N:
                    break
                parts = line.strip().split(',')
                if len(parts) > 2:
                    d = parts[2].strip().lower()
                    if d and d not in _HOSTING_PLATFORMS:
                        domains.add(d)
        if domains:
            print(f'[PhishingDetector] Popular domains loaded: {len(domains)} from Majestic', flush=True)
            return domains
    except Exception as exc:
        print(f'[PhishingDetector] Majestic fetch failed: {exc}', flush=True)

    # Tranco fallback
    domains = _fetch_tranco_domains()
    if domains:
        print(f'[PhishingDetector] Popular domains loaded: {len(domains)} from Tranco', flush=True)
    return domains


# Keep old name as alias so _popular_refresh_loop still works
_fetch_tranco = _fetch_popular_domains


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

    # ── Fix 2: Suppress meta_score for legitimate payment integrations ─────────
    # The DNN flags "form action → external domain" as suspicious (meta≈1.0),
    # but this is normal for PayPal/Stripe donation/checkout embeds.
    # If a recognised payment processor is present in the DOM, cap meta_score.
    if meta_score >= 0.8 and any(proc in dom.lower() for proc in _PAYMENT_PROCESSORS):
        meta_score = min(meta_score, 0.35)

    # ── Additional signal scorers ──────────────────────────────────────────────
    url_score       = _rule_url_score(url)
    _is_agency_site = _multi_brand_detected(url, text)
    brand_score     = 0.0 if _is_agency_site else _brand_text_score(url, text)
    gov_score       = _gov_impersonation_score(url)
    link_score      = _link_cluster_score(url, dom)
    payment_form_score = _payment_form_score(url, dom)  # Raw CC form without processor

    # ── Phase 1: WHOIS + dead-link + broken-link (parallel, no Gemini) ──────────
    # Run I/O checks first to get age_score, which determines whether Gemini
    # needs Google Search grounding (expensive, ~8-10s) or text-only (fast, ~3s).
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as _pool1:
        _f_age    = _pool1.submit(_domain_age_score, url)
        _f_dead   = _pool1.submit(_dead_link_score, url, dom)
        _f_broken = _pool1.submit(_broken_link_score, url, dom)

        age_score         = _f_age.result()
        dead_link_score   = _f_dead.result()
        broken_link_score = _f_broken.result()

    # ── Phase 2: Gemini brand check with age-aware grounding decision ─────────
    # New domains (age >= 0.8, i.e. < ~2 months old) are unlikely to be
    # legitimate established brands — skip grounding for speed.
    # Old/established domains (age < 0.8) may be real brands competing in the
    # same space as famous ones (e.g. keypal.pro vs Ledger) — use grounding to
    # verify legitimacy via Google Search.
    _run_gemini = brand_score == 0.0 and not _is_agency_site and (dom_score >= 0.6 or meta_score >= 0.6) and GEMINI_KEY
    if _run_gemini:
        _use_grounding = age_score < 0.8
        brand_score = _gemini_brand_check(url, text, use_grounding=_use_grounding)

    # NLP two-layer partner check: suppress false positives for legitimate dealers
    if link_score >= 0.8:
        if _is_brand_partner(dom):                                          # Layer 1
            link_score = 0.0
        elif _llm_partner_check(dom, _get_dominant_domain(url, dom)):      # Layer 2
            link_score = 0.0
    # ── Scenario detection ─────────────────────────────────────────────────────
    _scenarios: list[str] = []
    if brand_score >= 0.5:
        _scenarios.append('brand')
    if age_score >= 0.8 and (payment_form_score >= 0.4 or meta_score >= 0.6):
        _scenarios.append('new_financial')
    if len(_scenarios) >= 2:
        _scenario = 'high_risk'
    else:
        _scenario = _scenarios[0] if _scenarios else 'default'

    # ── Scenario weights ───────────────────────────────────────────────────────
    _W = {
        'default':       (0.20, 0.05, 0.20, 0.15, 0.20, 0.05, 0.05, 0.10),
        'brand':         (0.20, 0.05, 0.20, 0.15, 0.35, 0.02, 0.02, 0.01),
        'new_financial': (0.15, 0.25, 0.15, 0.15, 0.10, 0.08, 0.07, 0.05),
        'high_risk':     (0.15, 0.20, 0.15, 0.15, 0.15, 0.07, 0.06, 0.07),
    }
    _wu, _wa, _wd, _wm, _wb, _wdl, _wbl, _wp = _W[_scenario]

    # ── Fusion ─────────────────────────────────────────────────────────────────
    # Hard overrides — structural signals only.
    # dead_link re-added with DOM guard: SPAs (React Router) use href="#" giving
    # dead_link≈1, but legitimate SPAs have normal DOM structure (low dom_score).
    # Requiring dom >= 0.60 ensures only cloned pages trigger — not real SPAs.
    # broken_link_score excluded: Railway IP is blocked by many legitimate sites.
    # brand_score excluded: text keyword matches are too noisy to hard-override alone.
    _dead_link_override = dead_link_score >= 0.8 and dom_score >= 0.60
    if gov_score >= 0.8 or link_score >= 0.8 or payment_form_score >= 0.8 or _dead_link_override:
        final_score = max(gov_score, link_score, payment_form_score,
                          dead_link_score if _dead_link_override else 0.0)
    else:
        final_score = (
            _wu  * url_score          +
            _wa  * age_score          +
            _wd  * dom_score          +
            _wm  * meta_score         +
            _wb  * brand_score        +
            _wdl * dead_link_score    +
            _wbl * broken_link_score  +
            _wp  * payment_form_score
        )

    # ── Scenario overrides (floor) ─────────────────────────────────────────────
    # new_financial: new domain + dead links or payment form → at least suspicious
    if _scenario == 'new_financial' and (dead_link_score >= 0.4 or payment_form_score >= 0.4):
        final_score = max(final_score, 0.60)
    # Dead-clone floor: most/all links dead + phishing DOM structure → at least suspicious.
    # Covers aged/hijacked domains used for cloned phishing pages.
    # DOM threshold 0.35 guards against SPA false positives (React Router sites have
    # normal DOM structure, giving low dom_score even with all-dead href="#" links).
    if dead_link_score >= 0.8 and dom_score >= 0.35:
        final_score = max(final_score, 0.60)
    # high_risk: brand impersonation on a new financial domain → phishing
    if _scenario == 'high_risk':
        final_score = max(final_score, 0.75)

    # Triple-signal hard override: brand + meta + dom + url all high simultaneously
    # is a near-certain phishing indicator that fusion weights alone underweight.
    # url_score >= 0.10 guards against false positives on legitimate OAuth pages
    # (which have url_score ≈ 0 on clean domains).
    _triple_signal = (brand_score >= 0.8 and
                      meta_score  >= 0.8 and
                      dom_score   >= 0.8 and
                      url_score   >= 0.10)
    if _triple_signal:
        final_score = max(final_score, 0.85)

    # Brand + DOM override: Gemini-confirmed impersonation + phishing DOM structure
    # is sufficient for phishing even without meta/url signals.
    if brand_score >= 0.8 and dom_score >= 0.8:
        final_score = max(final_score, 0.85)

    # Adaptive threshold: suspicious URL lowers the bar for blocking.
    # When brand + metadata both fire at high confidence, tighten threshold:
    # these two signals together reliably indicate phishing even when DOM
    # score is slightly lower (e.g. SPA pages that render credentials lazily).
    _strong_signals = brand_score >= 0.8 and meta_score >= 0.8
    if _scenario in ('new_financial', 'high_risk'):
        _base = 0.60
    elif _strong_signals:
        _base = 0.70
    else:
        _base = 0.75
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
            'dead_link_factor':             round(dead_link_score,      4),
            'dead_link_message':            _dead_link_message(dead_link_score),
            'broken_link_factor':           round(broken_link_score,    4),
            'broken_link_message':          _broken_link_message(broken_link_score),
            'payment_form_factor':          round(payment_form_score,   4),
            'payment_form_message':         _payment_form_message(payment_form_score),
            'domain_age_factor':            round(age_score,    4),
            'domain_age_message':           _age_message(age_score),
            'scoring_scenario':             _scenario,
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
def _gov_impersonation_score(url: str) -> float:
    """
    Returns 1.0 if 'gov' appears as a hyphen-delimited word in the hostname
    but NOT in the structural TLD/SLD position that indicates a genuine
    government domain.

    Real government domains always have 'gov' (or 'go') at parts[-1] or
    parts[-2]:
      jpj.gov.my      → parts[-2] = 'gov' → safe
      example.gov     → parts[-1] = 'gov' → safe
      moe.go.id       → parts[-2] = 'go'  → safe
    Phishing domains put 'gov' in a prefix position:
      gov-jpj.evil.com   → parts[-2] = 'evil'  → phishing
      jpj-gov.evil.com   → parts[-2] = 'evil'  → phishing
      gov.phishing.com   → parts[-2] = 'phishing' → phishing
    """
    try:
        parsed = urlparse(url)
        host   = parsed.hostname or ''
        parts  = [p.lower() for p in host.split('.')]

        if len(parts) < 2:
            return 0.0

        # True government domain: 'gov'/'mil'/'go' is the TLD or SLD
        if parts[-1] in ('gov', 'mil'):
            return 0.0
        if parts[-2] in ('gov', 'go', 'mil'):
            return 0.0

        # 'gov' appears somewhere in the hostname but not in TLD/SLD → impersonation
        for part in parts:
            if 'gov' in part.split('-'):
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

# ── Payment form detection constants ──────────────────────────────────────────

# URL path segments that indicate a payment/checkout page
_CHECKOUT_PATHS = {'checkout', 'payment', 'pay', 'order', 'billing', 'purchase', 'cart'}

# Known legitimate payment processor domains/scripts
_PAYMENT_PROCESSORS = {
    'js.stripe.com', 'q.stripe.com',               # Stripe
    'paypal.com', 'paypalobjects.com',              # PayPal
    'braintreegateway.com', 'braintree-api.com',   # Braintree
    'squareup.com', 'square.com',                  # Square
    'adyen.com',                                    # Adyen
    'checkout.com',                                 # Checkout.com
    'mollie.com',                                   # Mollie
    '2checkout.com',                               # 2Checkout
    'authorize.net',                               # Authorize.Net
    'worldpay.com', 'cybersource.com',             # Worldpay / CyberSource
    'klarna.com', 'afterpay.com',                  # BNPL
    # Southeast Asia
    'billplz.com', 'toyyibpay.com', 'senangpay.my',
    'ipay88.com', 'razer.com', 'hitpay.com', 'curlec.com',
    'paydee.my', 'payex.com.my',
}

# Crypto wallet address input field patterns
_WALLET_INDICATORS = [
    r'placeholder=["\'][^"\']*(?:wallet\s*address|erc20|bep20|trc20|0x[0-9a-f]{4}|crypto\s*address)[^"\']*["\']',
    r'(?:name|id)=["\'][^"\']*(?:wallet[_\-]?addr|crypto[_\-]?addr|erc20|bep20)[^"\']*["\']',
    r'placeholder=["\'][^"\']*(?:enter\s*your\s*wallet|deposit\s*address|receiving\s*address)[^"\']*["\']',
]

# Input field patterns specific to credit card forms
_CC_INPUT_INDICATORS = [
    r'autocomplete=["\']cc-number["\']',
    r'(?:name|id)=["\'][^"\']*(?:card[_\-]?num|cc[_\-]?num|cardno|ccno|credit[_\-]?card)[^"\']*["\']',
    r'placeholder=["\'][^"\']*(?:card\s*number|\d{4}\s+\d{4})[^"\']*["\']',
]

_CVV_INDICATORS = [
    r'autocomplete=["\']cc-csc["\']',
    r'(?:name|id)=["\'][^"\']*(?:cvv|cvc|cvc2|security[_\-]?code|card[_\-]?code)[^"\']*["\']',
    r'placeholder=["\'][^"\']*(?:cvv|cvc\b|security\s*code)[^"\']*["\']',
]

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
            f'Is this page a LEGITIMATE website associated with {dominant_domain} '
            f'(e.g. an official product or service by the same company, a subsidiary, '
            f'an authorized partner/reseller, or a news/review/educational site about the brand), '
            f'or is it a FAKE site trying to deceive users into thinking '
            f'they are interacting with {dominant_domain}?\n\n'
            f'Reply with exactly one word: LEGITIMATE or IMPERSONATION.'
        )
        payload = json.dumps({
            'contents': [{'parts': [{'text': prompt}]}]
        }).encode()
        req = urllib.request.Request(
            f'https://generativelanguage.googleapis.com/v1beta/models/'
            f'gemini-3.1-flash-lite:generateContent?key={GEMINI_KEY}',
            data=payload,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
        answer = result['candidates'][0]['content']['parts'][0]['text'].strip().upper()
        print(f'[PhishingDetector] Gemini partner check → {answer} ({dominant_domain})')
        return 'LEGITIMATE' in answer
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


# ── 5b. HTTP link reachability scorer ─────────────────────────────────────────

def _broken_link_score(url: str, dom: str) -> float:
    """
    Samples up to 5 same-site links and checks whether they return valid HTTP
    responses. A cloned phishing page typically only wires up the credential
    form; other navigation links return 404/502 or a bare nginx/Apache default.

    Runs HEAD requests in parallel with a hard 3-second wall-clock cap.
    Returns the ratio of broken links if >= 50%, else 0.
    """
    if not dom:
        return 0.0
    try:
        soup = BeautifulSoup(dom, 'html.parser')
        parsed_base = urlparse(url)
        host   = parsed_base.hostname or ''
        base   = f"{parsed_base.scheme}://{host}"
        page_reg = _reg_domain(host)

        candidates: list[str] = []
        seen: set[str] = set()
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
            if href.startswith('/'):
                full = base + href
            elif href.startswith('http'):
                full = href
            else:
                continue
            link_host = urlparse(full).hostname or ''
            if _reg_domain(link_host) != page_reg:
                continue          # external link — skip
            norm = full.split('?')[0].rstrip('/')
            if norm in seen:
                continue
            seen.add(norm)
            candidates.append(full)
            if len(candidates) >= 5:
                break

        if len(candidates) < 3:
            return 0.0

        def _head(link_url: str) -> bool:
            """True if link is reachable (2xx / 3xx / 405 Method Not Allowed)."""
            try:
                req = urllib.request.Request(
                    link_url,
                    method='HEAD',
                    headers={'User-Agent': 'Mozilla/5.0 (compatible; PhishingDetector/1.0)'},
                )
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    return resp.status < 400
            except urllib.error.HTTPError as e:
                # 405 = server alive but rejects HEAD — count as reachable
                return e.code == 405 or e.code < 400
            except Exception:
                return False   # timeout, connection refused, etc.

        futures = [_link_check_pool.submit(_head, lnk) for lnk in candidates]
        ok = 0
        deadline = time.time() + 3.0
        try:
            for fut in concurrent.futures.as_completed(futures, timeout=max(0.1, deadline - time.time())):
                try:
                    if fut.result():
                        ok += 1
                except Exception:
                    pass
        except concurrent.futures.TimeoutError:
            pass

        total = len(candidates)
        ratio = (total - ok) / total
        print(f'[PhishingDetector] link-check {url}: {ok}/{total} reachable → broken={ratio:.2f}')
        return round(ratio, 4) if ratio >= 0.5 else 0.0

    except Exception:
        return 0.0


# ── 5c. Payment form scorer ───────────────────────────────────────────────────

def _payment_form_score(url: str, dom: str) -> float:
    """
    Detects pages that collect raw financial credentials without a recognized
    payment processor. Covers three attack types:
      (a) Credit card harvesting: card number + CVV fields
      (b) Crypto wallet harvesting: wallet address input fields

    Returns 0.8 for high-confidence matches, 0.4 for partial matches, 0.7 for
    crypto wallet inputs, 0.0 otherwise.
    """
    if not dom:
        return 0.0

    dom_lower = dom.lower()

    # ── Crypto wallet detection (independent of checkout path) ────────────────
    has_wallet = any(re.search(p, dom_lower) for p in _WALLET_INDICATORS)
    if has_wallet:
        return 0.7

    # ── Credit card detection ─────────────────────────────────────────────────
    # Trigger condition A: is this a payment-related page by URL?
    path_parts = set(urlparse(url).path.lower().strip('/').split('/'))
    is_checkout_page = bool(path_parts & _CHECKOUT_PATHS)

    # Trigger condition B: does DOM have CC input fields?
    has_cc  = any(re.search(p, dom_lower) for p in _CC_INPUT_INDICATORS)
    has_cvv = any(re.search(p, dom_lower) for p in _CVV_INDICATORS)

    # Skip entirely if neither condition is met
    if not (is_checkout_page or has_cc or has_cvv):
        return 0.0

    # Safe if a recognized payment processor script/iframe is present
    if any(proc in dom_lower for proc in _PAYMENT_PROCESSORS):
        return 0.0

    if has_cc and has_cvv:
        return 0.8    # Both card number + CVV fields with no processor → high risk
    if has_cc or has_cvv:
        return 0.4    # Only one type found → moderate risk
    # Checkout URL but no raw CC fields (e.g. redirects to hosted payment page)
    return 0.0


# ── 5. Brand text impersonation scorer ─────────────────────────────────────────

# Maps brand keywords to their legitimate domain suffixes.
_BRAND_DOMAINS: dict[str, list[str]] = {
    # 'paypal' removed: appears on any site accepting PayPal donations/payments.
    # Real PayPal phishing caught by dom/meta signals + Gemini grounding.
    'apple':           ['apple.com', 'icloud.com'],
    # 'google' removed from static keyword check — too many legitimate sites mention
    # "Google Review", "Google Maps", "Google Analytics" etc., causing false positives.
    # Real Google phishing pages are caught by dom/meta signals + Gemini grounding.
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
    'crypto.com':      ['crypto.com'],
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
    # Payment method brands removed from static check — these appear on any
    # legitimate e-commerce/donation site ("We accept Visa", "Pay with Klarna").
    # Keyword alone cannot distinguish acceptance from impersonation.
    # Real phishing for these brands is caught by dom/meta signals + Gemini.
    # 'visa', 'mastercard', 'american express', 'amex', 'western union',
    # 'wise', 'revolut', 'venmo', 'cash app', 'alipay', 'klarna',
    # 'afterpay', 'affirm'
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
    'crypto.com', 'steam', 'chime',
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


def _multi_brand_detected(url: str, text: str) -> bool:
    """
    Returns True if 3+ distinct brands appear in the page text.

    A page mentioning 3+ brands simultaneously cannot be impersonating all of
    them — it is almost certainly a partner/agency/comparison page listing the
    platforms it works with (e.g. "We build Shopify, WooCommerce, and Wix sites").
    In this case brand impersonation scoring should be suppressed entirely.
    """
    host       = urlparse(url).hostname or ''
    reg_domain = _reg_domain(host).lower()
    text_lower = text.lower()
    detected   = 0

    for brand, legit_domains in _BRAND_DOMAINS.items():
        pattern   = r'\b' + re.escape(brand) + r'\b'
        min_count = 2 if brand in _AMBIGUOUS_BRANDS else 1
        if len(re.findall(pattern, text_lower)) >= min_count:
            if not any(d in host for d in legit_domains):
                if re.search(pattern, reg_domain):
                    continue
                detected += 1
                if detected >= 3:
                    return True
    return False


# ── 5d. Gemini dynamic brand impersonation check ──────────────────────────────

def _gemini_brand_check(url: str, text: str, use_grounding: bool = True) -> float:
    """
    Asks Gemini whether the page is impersonating a real organization or brand.
    Covers fake login pages, fake stores, fake support, fake investment platforms,
    and any page that misrepresents its true owner or affiliation.

    Only called when:
      - Static _brand_text_score returned 0 (known brands not matched)
      - At least one other signal is elevated (dom >= 0.6 OR meta >= 0.6)
      - GEMINI_KEY is set

    Results cached 30 minutes to stay within free API quota.
    Returns 1.0 if impersonation detected, 0.0 otherwise.
    """
    if not GEMINI_KEY or not text.strip():
        return 0.0

    now = time.time()
    if url in _gemini_brand_cache:
        score, ts = _gemini_brand_cache[url]
        if now - ts < _GEMINI_BRAND_TTL:
            return score

    host       = urlparse(url).hostname or ''
    clean_text = ' '.join(text.split())[:600]

    prompt = (
        f'Use Google Search to look up information about the domain "{host}" '
        f'before answering.\n\n'
        f'Page text: "{clean_text}"\n'
        f'Domain: {host}\n\n'
        f'Is this page impersonating a DIFFERENT, more well-known brand or organization?\n\n'
        f'Important: If "{host}" is itself a legitimate brand, company, or official product '
        f'(even if lesser-known), reply NO — even if it operates in the same space as '
        f'famous brands (e.g. a real hardware wallet manufacturer is NOT impersonating '
        f'Ledger/Trezor; a real crypto exchange is NOT impersonating Binance).\n\n'
        f'These are also NOT impersonation:\n'
        f'- A page that merely mentions or discusses a brand (news, reviews, comparisons)\n'
        f'- A legitimate tool that relates to a brand '
        f'(e.g. a GPT-detection tool is not impersonating OpenAI)\n\n'
        f'Only reply YES if the page is actively deceiving users into thinking they are '
        f'on a DIFFERENT brand\'s official website or affiliated service.\n\n'
        f'If yes: YES [organization being impersonated]\n'
        f'If no: NO'
    )

    _GEMINI_BASE = 'https://generativelanguage.googleapis.com/v1beta/models/'

    def _call_gemini(grounding: bool) -> dict:
        payload = {'contents': [{'parts': [{'text': prompt}]}]}
        if grounding:
            payload['tools'] = [{'google_search': {}}]
            model, timeout = 'gemini-2.5-flash', 10
        else:
            model, timeout = 'gemini-3.1-flash-lite', 5
        req = urllib.request.Request(
            f'{_GEMINI_BASE}{model}:generateContent?key={GEMINI_KEY}',
            data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    try:
        try:
            result = _call_gemini(grounding=use_grounding)
            mode = 'grounded' if use_grounding else 'text-only'
            print(f'[PhishingDetector] Gemini brand check ({mode}) ({host})', flush=True)
        except urllib.error.HTTPError as e:
            if e.code == 429 and use_grounding:
                print(f'[PhishingDetector] Grounding 429, retrying text-only ({host})', flush=True)
                result = _call_gemini(grounding=False)
            else:
                raise
        except (TimeoutError, OSError):
            if use_grounding:
                print(f'[PhishingDetector] Grounding timeout, retrying text-only ({host})', flush=True)
                result = _call_gemini(grounding=False)
            else:
                raise
        answer = result['candidates'][0]['content']['parts'][0]['text'].strip()
        print(f'[PhishingDetector] Gemini brand check → {answer!r} ({host})', flush=True)

        if answer.upper().startswith('YES'):
            # Extract brand name from "YES [Koinly]" or "YES Koinly"
            match = re.search(r'YES\s*\[?([^\]\n,]+)', answer, re.IGNORECASE)
            if match:
                brand_name = match.group(1).strip().lower()
                # Safe if the brand name is part of the actual domain
                score = 0.0 if brand_name and brand_name in host.lower() else 1.0
            else:
                score = 1.0   # YES with no extractable name — still flag
        else:
            score = 0.0

    except Exception as exc:
        print(f'[PhishingDetector] Gemini brand check failed: {type(exc).__name__}: {exc}', flush=True)
        score = 0.0

    # Cache result (including failures) to avoid hammering the API on rate-limit errors
    _gemini_brand_cache[url] = (score, now)
    return score


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



# ── Diagnostic message generators ──────────────────────────────────────────────

def _broken_link_message(score: float) -> str:
    if score >= 0.8:
        return f'~{round(score * 100)}% of same-site links are unreachable — page is likely a cloned template.'
    if score >= 0.5:
        return f'~{round(score * 100)}% of same-site links returned errors — elevated suspicion.'
    return 'Same-site links appear reachable.'

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
        return 'Page identified as impersonating a real brand or organization.'
    return 'No brand impersonation detected.'

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

def _payment_form_message(score: float) -> str:
    if score >= 0.8:
        return 'Page collects raw credit card details without a recognized payment processor.'
    if score >= 0.7:
        return 'Page contains crypto wallet address input fields — possible wallet harvesting.'
    if score >= 0.4:
        return 'Page contains payment input fields without a recognized payment processor.'
    return 'No suspicious payment form detected.'


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
