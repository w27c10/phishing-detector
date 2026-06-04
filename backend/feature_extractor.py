"""
feature_extractor.py

Converts raw URL strings and DOM HTML into the three input tensors
expected by the multi-branch ONNX model:

  - URL tensor    : int32 (1, 256)  — character-level, printable-ASCII vocab
  - DOM tensor    : int32 (1, 500)  — HTML tag-name sequence
  - Meta tensor   : float32 (1, 20) — hand-crafted numerical features
"""

import math
import re
from urllib.parse import urlparse

import numpy as np
from bs4 import BeautifulSoup

# ── Constants ──────────────────────────────────────────────────────────────────

URL_MAX_LEN = 256
DOM_MAX_LEN = 500
META_DIM = 20

# Printable ASCII chars used in URLs, indexed 1..N (0 = padding / unknown).
_URL_CHARS = (
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '0123456789'
    '-._~:/?#[]@!$&\'()*+,;=%'
)
URL_VOCAB_SIZE = len(_URL_CHARS)  # 85
_CHAR_TO_IDX: dict[str, int] = {c: i + 1 for i, c in enumerate(_URL_CHARS)}

# HTML tags assigned a sequential index (0 = unknown / padding).
_DOM_TAGS = [
    'html', 'head', 'body', 'div', 'span', 'p', 'a', 'input', 'form', 'button',
    'script', 'style', 'link', 'img', 'table', 'tr', 'td', 'th', 'ul', 'ol',
    'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'nav', 'header', 'footer',
    'section', 'article', 'aside', 'main', 'iframe', 'embed', 'object', 'meta',
    'title', 'label', 'select', 'option', 'textarea', 'fieldset', 'legend',
    'br', 'hr', 'strong', 'em', 'b', 'i', 'pre', 'code', 'blockquote',
    'canvas', 'video', 'audio', 'source', 'track', 'svg', 'path',
]
DOM_VOCAB_SIZE = len(_DOM_TAGS)  # 61
_TAG_TO_IDX: dict[str, int] = {t: i + 1 for i, t in enumerate(_DOM_TAGS)}


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_url_features(url: str) -> np.ndarray:
    """Returns int32 array of shape (1, URL_MAX_LEN)."""
    seq = [_CHAR_TO_IDX.get(c, 0) for c in url[:URL_MAX_LEN]]
    seq += [0] * (URL_MAX_LEN - len(seq))
    return np.array([seq], dtype=np.int32)


def extract_dom_features(html: str) -> np.ndarray:
    """Returns int32 array of shape (1, DOM_MAX_LEN)."""
    try:
        if not html or '<' not in html:
            tags = []
        else:
            soup = BeautifulSoup(html, 'html.parser')
            tags = [tag.name for tag in soup.find_all() if tag.name]
    except Exception:
        tags = []
    seq = [_TAG_TO_IDX.get(t, 0) for t in tags[:DOM_MAX_LEN]]
    seq += [0] * (DOM_MAX_LEN - len(seq))
    return np.array([seq], dtype=np.int32)


def extract_metadata_features(url: str, html: str) -> np.ndarray:
    """Returns float32 array of shape (1, META_DIM=20)."""
    feats = _url_meta(url) + _dom_meta(url, html)
    assert len(feats) == META_DIM, f"Expected {META_DIM} features, got {len(feats)}"
    return np.array([feats], dtype=np.float32)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _url_meta(url: str) -> list[float]:
    """10 URL-level features, all normalised to [0, 1]."""
    parsed = urlparse(url)
    host = parsed.hostname or ''

    return [
        min(len(url) / 200.0, 1.0),                                    # 1  url_length_norm
        min(url.count('.') / 10.0, 1.0),                                # 2  dot_count_norm
        min(url.count('-') / 10.0, 1.0),                                # 3  hyphen_count_norm
        float('@' in url),                                               # 4  has_at_symbol
        min(len(host.split('.')) / 5.0, 1.0),                           # 5  subdomain_depth_norm
        float(bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host))),       # 6  is_ip_address
        float(parsed.scheme == 'https'),                                 # 7  is_https
        min(_entropy(url) / 6.0, 1.0),                                  # 8  url_entropy_norm
        sum(c.isdigit() for c in host) / max(len(host), 1),             # 9  digit_ratio_in_host
        float(bool(parsed.port)),                                        # 10 has_non_default_port
    ]


def _dom_meta(url: str, html: str) -> list[float]:
    """10 DOM-level features, all normalised to [0, 1]."""
    try:
        if not html or '<' not in html:
            return [0.0] * 10
        soup = BeautifulSoup(html, 'html.parser')
    except Exception:
        return [0.0] * 10

    base_domain = urlparse(url).hostname or ''

    all_tags   = soup.find_all()
    inputs     = soup.find_all('input')
    pw_inputs  = soup.find_all('input', {'type': 'password'})
    all_links  = soup.find_all('a', href=True)
    ext_links  = [a for a in all_links if _is_external(a['href'], base_domain)]
    scripts    = soup.find_all('script')
    ext_scripts = [s for s in scripts if s.get('src') and _is_external(s['src'], base_domain)]
    iframes    = soup.find_all('iframe')
    forms      = soup.find_all('form')
    hidden     = soup.find_all(style=re.compile(r'display\s*:\s*none', re.I))

    ext_form_action = any(
        _is_external(f.get('action', ''), base_domain) for f in forms
    )
    has_favicon = bool(
        soup.find('link', rel=lambda r: isinstance(r, list) and 'icon' in r)
    )

    return [
        min(len(all_tags)    / 500.0, 1.0),                             # 11 tag_count_norm
        min(len(inputs)      / 10.0,  1.0),                             # 12 input_count_norm
        float(len(pw_inputs) > 0),                                       # 13 has_password_field
        len(ext_links) / max(len(all_links), 1),                        # 14 ext_link_ratio
        min(len(scripts)     / 10.0,  1.0),                             # 15 script_count_norm
        len(ext_scripts) / max(len(scripts), 1),                        # 16 ext_script_ratio
        min(len(iframes)     / 5.0,   1.0),                             # 17 iframe_count_norm
        float(has_favicon),                                              # 18 has_favicon
        float(ext_form_action),                                          # 19 form_action_external
        min(len(hidden)      / 10.0,  1.0),                             # 20 hidden_element_norm
    ]


def _is_external(href: str, base_domain: str) -> bool:
    if not href or href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
        return False
    if href.startswith('/'):
        return False
    netloc = urlparse(href).netloc
    if not netloc:
        return False
    return base_domain not in netloc


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return -sum((f / n) * math.log2(f / n) for f in freq.values())
