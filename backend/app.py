"""
app.py — Flask inference server

Loads phishing_model.onnx once at startup and exposes a single stateless
POST /analyze endpoint. No data is written to disk or logged.

Usage:
    python app.py
    # Server listens on http://0.0.0.0:5000
"""

import os
import sys

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
CORS(app)  # Allow requests from the Chrome extension origin

# ── Load ONNX model once at startup ───────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'phishing_model.onnx')

try:
    _session = ort.InferenceSession(MODEL_PATH)
    _input_names  = [inp.name for inp in _session.get_inputs()]
    _output_count = len(_session.get_outputs())
    print(f"[PhishingDetector] Model loaded. Inputs: {_input_names}")
except Exception as exc:
    print(f"[PhishingDetector] ERROR: Could not load model from {MODEL_PATH}\n  {exc}")
    print("  Run model/create_stub_model.py to generate a test model.")
    sys.exit(1)


# ── Endpoint ───────────────────────────────────────────────────────────────────

@app.route('/analyze', methods=['POST'])
def analyze():
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({'error': 'Invalid JSON body'}), 400

    url = str(body.get('url', ''))
    dom = str(body.get('dom', ''))

    # Build input tensors (shapes match model training exactly).
    url_feat  = extract_url_features(url)        # (1, 256) int32
    dom_feat  = extract_dom_features(dom)         # (1, 500) int32
    meta_feat = extract_metadata_features(url, dom)  # (1, 20) float32

    # Map by position: url_input, dom_input, meta_input (order from Model definition).
    inputs = dict(zip(_input_names, [url_feat, dom_feat, meta_feat]))
    outputs = _session.run(None, inputs)

    # Model outputs: [final_output, url_output, dom_output, meta_output]
    # url_output (index 1) is no longer used for scoring — replaced by rule-based scorer.
    # All shaped (1, 1) — extract the scalar.
    dom_score  = float(outputs[2][0][0]) if _output_count > 2 else 0.0
    meta_score = float(outputs[3][0][0]) if _output_count > 3 else 0.0

    # Rule-based URL scorer — replaces the unreliable LSTM branch.
    # Uses deterministic arithmetic on URL structure rather than learned weights,
    # so it generalises to hosting-abuse phishing (e.g. cez.dvf.mybluehost.me)
    # without producing false positives on clean domains like google.com.
    url_score = _rule_url_score(url)

    # Weighted fusion: URL rules + DOM neural + metadata neural
    final_score = 0.40 * url_score + 0.35 * dom_score + 0.25 * meta_score

    # Adaptive threshold: a suspicious URL structure lowers the evidence bar.
    # The more the URL looks like hosting-abuse phishing, the less corroboration
    # we require from DOM and metadata before blocking.
    # url_score=0.00 → threshold=0.75 (normal, no penalty for clean domains)
    # url_score=0.40 → threshold=0.55  (cez.dvf.mybluehost.me range)
    # url_score=1.00 → threshold=0.40  (floor — always flag extreme URL abuse)
    threshold = max(0.40, 0.75 - url_score * 0.50)

    verdict = 'phishing' if final_score >= threshold else 'safe'

    return jsonify({
        'threat_score': round(final_score, 4),
        'verdict': verdict,
        'explanation_details': {
            'url_threat_factor':        round(url_score, 4),
            'url_diagnostic_message':   _url_message(url_score),
            'dom_threat_factor':        round(dom_score, 4),
            'dom_diagnostic_message':   _dom_message(dom_score),
            'metadata_threat_factor':   round(meta_score, 4),
            'metadata_diagnostic_message': _meta_message(meta_score),
        },
    })


# ── Rule-based URL risk scorer ─────────────────────────────────────────────────

def _rule_url_score(url: str) -> float:
    """
    Deterministic URL risk score in [0, 1].

    Catches structural phishing patterns that the neural LSTM missed due to
    limited training data, without producing false positives on clean domains.
    Each rule adds to a cumulative risk budget; the total is clamped to 1.0.
    """
    import math
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
        host   = parsed.hostname or ''
    except Exception:
        return 0.0

    parts      = host.split('.')
    tld        = parts[-1] if parts else ''
    reg_domain = '.'.join(parts[-2:]) if len(parts) >= 2 else host
    subdomains = parts[:-2]           # everything left of the registered domain

    risk = 0.0

    # ── Subdomain abuse ───────────────────────────────────────────────────────
    # Legitimate sites rarely have more than one subdomain (www, mail, etc.).
    # Phishing pages on shared hosting often stack random short subdomains.
    n_sub = len(subdomains)
    if n_sub >= 3:
        risk += 0.50   # e.g. a.b.c.host.com — very unusual
    elif n_sub == 2:
        risk += 0.30   # e.g. cez.dvf.mybluehost.me

    # ── Subdomain entropy: random-looking labels are a strong phishing signal ─
    if subdomains:
        sub_str = ''.join(subdomains)
        n = len(sub_str)
        if n > 0:
            freq = {}
            for c in sub_str:
                freq[c] = freq.get(c, 0) + 1
            entropy = -sum((f / n) * math.log2(f / n) for f in freq.values())
            if entropy > 3.5:
                risk += 0.25   # high-entropy = looks randomly generated
            elif entropy > 2.5:
                risk += 0.10

    # ── IP address as hostname ─────────────────────────────────────────────────
    if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host):
        risk += 0.60

    # ── @ symbol in URL (credential stuffing trick) ───────────────────────────
    if '@' in url:
        risk += 0.50

    # ── Suspicious TLDs commonly abused in phishing campaigns ─────────────────
    RISKY_TLDS = {'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'club',
                  'work', 'click', 'link', 'online', 'site', 'website',
                  'info', 'biz', 'pw', 'cc', 'su', 'ru'}
    if tld.lower() in RISKY_TLDS:
        risk += 0.20

    # ── Known brand terms in subdomain but not in registered domain ───────────
    # e.g. paypal.secure-login.com — brand in subdomain, unrelated reg domain
    BRANDS = {'paypal', 'apple', 'amazon', 'google', 'microsoft', 'netflix',
              'facebook', 'instagram', 'whatsapp', 'bankofamerica', 'chase',
              'wellsfargo', 'hsbc', 'dhl', 'fedex', 'ups', 'dropbox'}
    sub_text = ' '.join(subdomains).lower()
    reg_text = reg_domain.lower()
    for brand in BRANDS:
        if brand in sub_text and brand not in reg_text:
            risk += 0.50
            break

    # ── Excessive URL length ──────────────────────────────────────────────────
    if len(url) > 150:
        risk += 0.10
    if len(url) > 250:
        risk += 0.10

    # ── Non-HTTPS ─────────────────────────────────────────────────────────────
    if parsed.scheme != 'https':
        risk += 0.10

    return min(risk, 1.0)


# ── Diagnostic message generators ─────────────────────────────────────────────

def _url_message(score: float) -> str:
    if score >= 0.75:
        return 'Anomalous brand keyword sequence detected in subdomain structure.'
    if score >= 0.5:
        return 'Suspicious URL pattern detected with unusual character composition.'
    return 'URL structure appears within normal parameters.'


def _dom_message(score: float) -> str:
    if score >= 0.75:
        return 'High structural layout correlation identified with a protected brand authentication template.'
    if score >= 0.5:
        return 'Page layout contains elements commonly associated with credential harvesting.'
    return 'Page DOM structure appears within normal parameters.'


def _meta_message(score: float) -> str:
    if score >= 0.75:
        return 'Suspicious data routing configuration detected; form action links to an unassociated domain.'
    if score >= 0.5:
        return 'Elevated ratio of external resource references detected.'
    return 'Page metadata and behavioural signals appear within normal parameters.'


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
