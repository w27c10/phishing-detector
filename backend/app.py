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
    # All shaped (1, 1) — extract the scalar.
    url_score  = float(outputs[1][0][0]) if _output_count > 1 else 0.0
    dom_score  = float(outputs[2][0][0]) if _output_count > 2 else 0.0
    meta_score = float(outputs[3][0][0]) if _output_count > 3 else 0.0

    # URL branch excluded from verdict — it produces too many false positives
    # due to insufficient training coverage of legitimate URL patterns.
    # DOM structure and metadata behavioural signals are more reliable.
    final_score = 0.5 * dom_score + 0.5 * meta_score

    verdict = 'phishing' if final_score >= 0.75 else 'safe'

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
