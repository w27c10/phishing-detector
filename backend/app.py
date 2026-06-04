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
import concurrent.futures
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlparse

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


# ── Endpoint ───────────────────────────────────────────────────────────────────

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

    # ── Neural branch inference ────────────────────────────────────────────────
    url_feat  = extract_url_features(url)
    dom_feat  = extract_dom_features(dom)
    meta_feat = extract_metadata_features(url, dom)

    inputs  = dict(zip(_input_names, [url_feat, dom_feat, meta_feat]))
    outputs = _session.run(None, inputs)

    dom_score  = float(outputs[2][0][0]) if _output_count > 2 else 0.0
    meta_score = float(outputs[3][0][0]) if _output_count > 3 else 0.0

    # ── Additional signal scorers ──────────────────────────────────────────────
    url_score    = _rule_url_score(url)
    brand_score  = _brand_text_score(url, text)
    gov_score    = _gov_impersonation_score(url)
    link_score   = _link_cluster_score(url, dom)
    age_score    = _domain_age_score(url)        # WHOIS, cached + timeout
    visual_score = _visual_score(url, screenshot)

    # ── Fusion ─────────────────────────────────────────────────────────────────
    # Hard overrides — any one firing confidently means phishing.
    if brand_score >= 0.8 or visual_score >= 0.7 or gov_score >= 0.8 or link_score >= 0.8:
        final_score = max(brand_score, visual_score, gov_score, link_score)
    else:
        final_score = (
            0.25 * url_score   +
            0.20 * age_score   +
            0.20 * dom_score   +
            0.15 * meta_score  +
            0.10 * brand_score +
            0.10 * visual_score
        )

    # Adaptive threshold: suspicious URL lowers the bar for blocking
    threshold = max(0.40, 0.75 - url_score * 0.50)
    verdict   = 'phishing' if final_score >= threshold else 'safe'

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
            'link_cluster_factor':          round(link_score,   4),
            'link_cluster_message':         _link_cluster_message(link_score),
            'domain_age_factor':            round(age_score,    4),
            'domain_age_message':           _age_message(age_score),
            'visual_threat_factor':         round(visual_score, 4),
            'visual_diagnostic_message':    _visual_message(visual_score),
        },
    })


# ── 1. Rule-based URL risk scorer ──────────────────────────────────────────────

def _rule_url_score(url: str) -> float:
    """Deterministic URL risk score [0, 1] based on structural patterns."""
    try:
        parsed = urlparse(url)
        host   = parsed.hostname or ''
    except Exception:
        return 0.0

    parts      = host.split('.')
    tld        = parts[-1] if parts else ''
    reg_domain = '.'.join(parts[-2:]) if len(parts) >= 2 else host
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
    BRANDS = {'paypal','apple','amazon','google','microsoft','netflix',
              'facebook','instagram','whatsapp','bankofamerica','chase',
              'wellsfargo','hsbc','dhl','fedex','ups','usps','dropbox'}
    sub_text = ' '.join(subdomains).lower()
    reg_text = reg_domain.lower()
    for brand in BRANDS:
        if brand in sub_text and brand not in reg_text:
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

        external_domains: list[str] = []
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
            netloc = urlparse(href).netloc
            if not netloc or host in netloc:
                continue                              # internal / relative link
            parts  = netloc.split('.')
            reg    = '.'.join(parts[-2:]) if len(parts) >= 2 else netloc
            external_domains.append(reg.lower())

        if len(external_domains) < 5:
            return 0.0

        top_domain, top_count = Counter(external_domains).most_common(1)[0]
        ratio = top_count / len(external_domains)

        # >80 % of all external links pointing to one domain = strong clone signal
        return round(ratio, 4) if ratio >= 0.80 else 0.0
    except Exception:
        return 0.0


# ── 4. Brand text impersonation scorer ─────────────────────────────────────────

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
    'coinbase':        ['coinbase.com'],
    'binance':         ['binance.com'],
    'crypto':          ['crypto.com'],
}

def _brand_text_score(url: str, text: str) -> float:
    """
    Returns 1.0 if a recognised brand name appears in the page's visible text
    but the URL does not belong to that brand's legitimate domain.
    """
    host       = urlparse(url).hostname or ''
    text_lower = text.lower()

    for brand, legit_domains in _BRAND_DOMAINS.items():
        if brand in text_lower:
            if not any(d in host for d in legit_domains):
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
    'netflix':   [(229, 9, 20),   (20, 20, 20)],
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
                if any(_colour_dist(p, c) < 40 for c in colours)
            )
            coverage = matched / n
            if coverage > 0.18:   # brand colour covers >18 % of the viewport
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
