"""
collect_dataset.py

Builds a labeled phishing-detection dataset by:
  1. Fetching live phishing URLs from OpenPhish (free, no auth) and
     optionally PhishTank (requires a free API key).
  2. Fetching legitimate URLs from the Tranco top-1M list.
  3. Crawling the HTML of each URL (short timeout, privacy-safe).
  4. Saving dataset.csv — compatible with train_model.py (url, dom, label).

The output CSV mimics the Kaggle "Web Page Phishing Detection" style:
every row carries the raw url and dom alongside pre-computed URL-level
feature columns so the file is useful for classical ML baselines too.

Usage:
    pip install requests pandas tqdm beautifulsoup4 lxml
    python collect_dataset.py

Optional (PhishTank gives extra verified samples):
    python collect_dataset.py --phishtank-key YOUR_API_KEY

Output:
    dataset.csv          — train_model.py-ready (url, dom, label)
    dataset_features.csv — same rows with pre-computed feature columns added
"""

import argparse
import csv
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import pandas as pd
import requests
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────

MAX_PHISHING   = 3000   # cap on phishing samples fetched
MAX_LEGITIMATE = 3000   # cap on legitimate samples fetched
CRAWL_TIMEOUT  = 6      # seconds per page
MAX_DOM_BYTES  = 500_000
CRAWL_WORKERS  = 20     # parallel HTTP threads

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
}

# Well-known legitimate domains — used if both Tranco and Majestic fail.
# Covers a broad mix of categories to reduce dataset bias.
FALLBACK_LEGITIMATE = [
    # Search / Portals
    'https://www.google.com',      'https://www.bing.com',
    'https://www.yahoo.com',       'https://www.baidu.com',
    'https://www.duckduckgo.com',  'https://www.yandex.com',
    # Social
    'https://www.facebook.com',    'https://www.twitter.com',
    'https://www.instagram.com',   'https://www.linkedin.com',
    'https://www.reddit.com',      'https://www.pinterest.com',
    'https://www.tiktok.com',      'https://www.tumblr.com',
    'https://www.discord.com',     'https://www.twitch.tv',
    'https://www.snapchat.com',    'https://www.telegram.org',
    # Video / Media
    'https://www.youtube.com',     'https://www.netflix.com',
    'https://www.spotify.com',     'https://www.vimeo.com',
    'https://www.soundcloud.com',  'https://www.hulu.com',
    'https://www.bbc.com',         'https://www.cnn.com',
    'https://www.nytimes.com',     'https://www.theguardian.com',
    'https://www.reuters.com',     'https://www.apnews.com',
    # E-commerce
    'https://www.amazon.com',      'https://www.ebay.com',
    'https://www.etsy.com',        'https://www.shopify.com',
    'https://www.walmart.com',     'https://www.aliexpress.com',
    'https://www.target.com',      'https://www.bestbuy.com',
    # Finance / Payments
    'https://www.paypal.com',      'https://www.stripe.com',
    'https://www.chase.com',       'https://www.bankofamerica.com',
    'https://www.wellsfargo.com',  'https://www.coinbase.com',
    # Tech / Dev
    'https://www.microsoft.com',   'https://www.apple.com',
    'https://www.github.com',      'https://www.stackoverflow.com',
    'https://www.mozilla.org',     'https://www.cloudflare.com',
    'https://www.aws.amazon.com',  'https://www.azure.microsoft.com',
    'https://www.developers.google.com', 'https://www.npmjs.com',
    'https://www.pypi.org',        'https://www.docker.com',
    'https://www.heroku.com',      'https://www.digitalocean.com',
    # Productivity / Cloud
    'https://www.office.com',      'https://www.outlook.com',
    'https://www.drive.google.com','https://www.docs.google.com',
    'https://www.dropbox.com',     'https://www.notion.so',
    'https://www.slack.com',       'https://www.zoom.us',
    'https://www.trello.com',      'https://www.asana.com',
    'https://www.salesforce.com',  'https://www.hubspot.com',
    # Education
    'https://www.wikipedia.org',   'https://www.khanacademy.org',
    'https://www.coursera.org',    'https://www.edx.org',
    'https://www.udemy.com',       'https://www.medium.com',
    'https://www.quora.com',       'https://www.arxiv.org',
    # Government / Non-profit
    'https://www.nasa.gov',        'https://www.who.int',
    'https://www.un.org',          'https://www.nih.gov',
    'https://www.cdc.gov',         'https://www.europa.eu',
    # Travel / Maps
    'https://www.booking.com',     'https://www.airbnb.com',
    'https://www.expedia.com',     'https://www.tripadvisor.com',
    'https://maps.google.com',     'https://www.openstreetmap.org',
    # Misc popular
    'https://www.adobe.com',       'https://www.wordpress.org',
    'https://www.blogger.com',     'https://www.medium.com',
    'https://www.zendesk.com',     'https://www.mailchimp.com',
    'https://www.squarespace.com', 'https://www.wix.com',
]


# ── URL fetching ───────────────────────────────────────────────────────────────

def fetch_openphish() -> list[str]:
    """Free OpenPhish feed — one URL per line, no auth required."""
    print('[collect] Fetching OpenPhish feed...')
    try:
        r = requests.get('https://openphish.com/feed.txt', timeout=15, headers=HEADERS)
        r.raise_for_status()
        urls = [l.strip() for l in r.text.splitlines() if l.strip().startswith('http')]
        print(f'[collect] OpenPhish: {len(urls)} URLs')
        return urls
    except Exception as e:
        print(f'[collect] OpenPhish failed: {e}')
        return []


def fetch_phishtank(api_key: str = '') -> list[str]:
    """PhishTank CSV download.

    Without an API key, tries the anonymous endpoint (may be rate-limited).
    Register free at https://phishtank.org/register.php to get a key.
    """
    print('[collect] Fetching PhishTank CSV...')
    # Anonymous endpoint works without a key but has a lower rate limit.
    url = (
        f'https://data.phishtank.com/data/{api_key}/online-valid.csv'
        if api_key
        else 'https://data.phishtank.com/data/online-valid.csv'
    )
    try:
        r = requests.get(url, timeout=60, headers=HEADERS, stream=True)
        r.raise_for_status()

        # Response may be very large — parse line by line to avoid loading
        # the entire file into memory.
        urls = []
        first = True
        header = []
        for raw in r.iter_lines(decode_unicode=True):
            line = raw.strip()
            if not line:
                continue
            if first:
                header = [h.strip() for h in line.split(',')]
                first = False
                continue
            # Split carefully — URL field may contain commas if quoted
            parts = list(csv.reader([line]))[0]
            if not parts:
                continue
            row = dict(zip(header, parts))
            u = row.get('url', '').strip()
            if u.startswith('http'):
                urls.append(u)
            if len(urls) >= MAX_PHISHING:
                break

        print(f'[collect] PhishTank: {len(urls)} URLs')
        return urls
    except Exception as e:
        print(f'[collect] PhishTank failed: {e}')
        return []


def fetch_legitimate(n: int) -> list[str]:
    """Fetch n legitimate domain URLs — tries Majestic Million then fallback."""
    return fetch_majestic(n)


def fetch_majestic(n: int) -> list[str]:
    """Majestic Million — free CSV, columns: GlobalRank,...,Domain,..."""
    print(f'[collect] Fetching Majestic Million top-{n}...')
    try:
        r = requests.get(
            'https://downloads.majestic.com/majestic_million.csv',
            timeout=30, headers=HEADERS, stream=True,
        )
        r.raise_for_status()
        urls = []
        for i, line in enumerate(r.iter_lines(decode_unicode=True)):
            if i == 0:
                continue  # header
            if len(urls) >= n:
                break
            parts = line.split(',')
            if len(parts) >= 3:
                domain = parts[2].strip()
                if domain:
                    urls.append(f'https://{domain}')
        print(f'[collect] Majestic: {len(urls)} domains')
        return urls
    except Exception as e:
        print(f'[collect] Majestic failed ({e}), using hardcoded fallback list')
        return FALLBACK_LEGITIMATE


# ── HTML crawling ──────────────────────────────────────────────────────────────

def crawl(url: str) -> str:
    """Fetch HTML for a URL. Returns empty string on any failure."""
    try:
        r = requests.get(
            url,
            timeout=CRAWL_TIMEOUT,
            headers=HEADERS,
            allow_redirects=True,
            stream=True,
        )
        content_type = r.headers.get('Content-Type', '')
        if 'text/html' not in content_type and 'text/plain' not in content_type:
            return ''
        # Read up to MAX_DOM_BYTES
        chunks = []
        total = 0
        for chunk in r.iter_content(chunk_size=8192, decode_unicode=True):
            if isinstance(chunk, bytes):
                chunk = chunk.decode('utf-8', errors='ignore')
            chunks.append(chunk)
            total += len(chunk)
            if total >= MAX_DOM_BYTES:
                break
        return ''.join(chunks)
    except Exception:
        return ''


def crawl_batch(urls: list[str], label: int, desc: str) -> list[dict]:
    rows = []
    with ThreadPoolExecutor(max_workers=CRAWL_WORKERS) as ex:
        future_to_url = {ex.submit(crawl, u): u for u in urls}
        for future in tqdm(as_completed(future_to_url), total=len(urls), desc=desc, ncols=80):
            u = future_to_url[future]
            dom = future.result()
            rows.append({'url': u, 'dom': dom, 'label': label})
    return rows


# ── Feature computation (Kaggle-style extra columns) ──────────────────────────

def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return -sum((f / n) * math.log2(f / n) for f in freq.values())


def url_features(url: str) -> dict:
    """Pre-compute URL-level features (mirrors Kaggle dataset columns)."""
    parsed = urlparse(url)
    host   = parsed.hostname or ''
    path   = parsed.path   or ''

    return {
        'length_url':         len(url),
        'length_hostname':    len(host),
        'nb_dots':            url.count('.'),
        'nb_hyphens':         url.count('-'),
        'nb_at':              url.count('@'),
        'nb_qm':              url.count('?'),
        'nb_and':             url.count('&'),
        'nb_eq':              url.count('='),
        'nb_underscore':      url.count('_'),
        'nb_percent':         url.count('%'),
        'nb_slash':           url.count('/'),
        'nb_colon':           url.count(':'),
        'nb_semicolon':       url.count(';'),
        'nb_www':             int('www.' in url.lower()),
        'nb_com':             int('.com' in url.lower()),
        'https_token':        int(url.lower().startswith('https')),
        'ratio_digits_url':   sum(c.isdigit() for c in url) / max(len(url), 1),
        'ratio_digits_host':  sum(c.isdigit() for c in host) / max(len(host), 1),
        'nb_subdomains':      max(len(host.split('.')) - 2, 0),
        'prefix_suffix':      int('-' in host),
        'is_ip':              int(bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host))),
        'has_port':           int(bool(parsed.port)),
        'url_entropy':        round(_entropy(url), 4),
        'length_path':        len(path),
    }


def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    feat_rows = [url_features(u) for u in df['url']]
    feat_df = pd.DataFrame(feat_rows)
    return pd.concat([df.reset_index(drop=True), feat_df], axis=1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phishtank-key', default='', help='PhishTank API key (optional — anonymous access used if omitted)')
    parser.add_argument('--no-crawl', action='store_true', help='Skip HTML crawling (URL+metadata only)')
    parser.add_argument('--out', default='dataset.csv', help='Output CSV path')
    args = parser.parse_args()

    # ── 1. Collect phishing URLs ───────────────────────────────────────────────
    phishing_urls = fetch_phishtank(args.phishtank_key)   # PhishTank first (larger)
    phishing_urls += fetch_openphish()                     # OpenPhish supplement
    phishing_urls = list(dict.fromkeys(phishing_urls))[:MAX_PHISHING]  # deduplicate + cap

    if not phishing_urls:
        print('[collect] ERROR: No phishing URLs collected. Check your internet connection.')
        sys.exit(1)

    # ── 2. Collect legitimate URLs ─────────────────────────────────────────────
    # Fetch 2× the target to absorb crawl failures and still hit the cap.
    target_legit = len(phishing_urls)  # aim for 1:1 balance
    legitimate_urls = fetch_legitimate(target_legit * 2)
    legitimate_urls = legitimate_urls[:target_legit * 2]  # crawl extras, trim after

    # ── 3. Crawl HTML ─────────────────────────────────────────────────────────
    if args.no_crawl:
        print('[collect] Skipping HTML crawl (--no-crawl)')
        phishing_rows   = [{'url': u, 'dom': '', 'label': 1} for u in phishing_urls]
        legitimate_rows = [{'url': u, 'dom': '', 'label': 0} for u in legitimate_urls]
    else:
        print(f'\n[collect] Crawling {len(phishing_urls)} phishing URLs...')
        phishing_rows = crawl_batch(phishing_urls, label=1, desc='Phishing')

        print(f'\n[collect] Crawling {len(legitimate_urls)} legitimate URLs...')
        legitimate_rows = crawl_batch(legitimate_urls, label=0, desc='Legitimate')

    # ── 4. Assemble and balance ────────────────────────────────────────────────
    all_rows = phishing_rows + legitimate_rows
    df = pd.DataFrame(all_rows, columns=['url', 'dom', 'label'])

    # Drop rows with no URL
    df = df[df['url'].str.startswith('http')].reset_index(drop=True)

    # Balance: cap the majority class to match the minority class size.
    n_phish = int(df['label'].sum())
    n_safe  = len(df) - n_phish
    minority = min(n_phish, n_safe)
    df = pd.concat([
        df[df['label'] == 1].sample(n=min(n_phish, minority), random_state=42),
        df[df['label'] == 0].sample(n=min(n_safe,  minority), random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    n_phish = int(df['label'].sum())
    n_safe  = len(df) - n_phish

    # Stats
    n_phish = int(df['label'].sum())
    n_safe  = len(df) - n_phish
    n_with_dom = int((df['dom'].str.len() > 0).sum())
    print(f'\n[collect] Dataset summary:')
    print(f'  Total rows  : {len(df)}')
    print(f'  Phishing    : {n_phish}')
    print(f'  Legitimate  : {n_safe}')
    print(f'  With HTML   : {n_with_dom}  ({n_with_dom*100//max(len(df),1)}%)')

    # Save train_model.py-compatible CSV
    out_path = args.out
    df[['url', 'dom', 'label']].to_csv(out_path, index=False, escapechar='\\')
    print(f'[collect] Saved → {out_path}')

    # Save feature-enriched CSV (Kaggle-style extra columns)
    feat_path = out_path.replace('.csv', '_features.csv')
    df_feat = add_feature_columns(df[['url', 'dom', 'label']])
    df_feat.to_csv(feat_path, index=False, escapechar='\\')
    print(f'[collect] Saved → {feat_path}')


if __name__ == '__main__':
    main()
