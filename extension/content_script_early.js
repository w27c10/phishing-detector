/**
 * Early Content Script — document_start
 *
 * Runs before any page JS executes. Goals:
 *   1. Capture the original URL for redirect tracking (stored in background).
 *   2. Freeze the page on high-risk local signals while backend check runs.
 *   3. Hard-block (inject overlay) if backend confirms phishing with high confidence.
 *
 * Does NOT scrape DOM — nothing is available yet at document_start.
 */
(function () {
  'use strict';

  const url = window.location.href;
  if (!url.startsWith('http://') && !url.startsWith('https://')) return;

  // ── Local trusted domains (zero-latency bypass) ──────────────────────────
  const TRUSTED_DOMAINS = new Set([
    'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'x.com',
    'instagram.com', 'linkedin.com', 'github.com', 'apple.com', 'microsoft.com',
    'amazon.com', 'netflix.com', 'wikipedia.org', 'reddit.com', 'bing.com',
    'yahoo.com', 'whatsapp.com', 'telegram.org',
  ]);

  let hostname;
  try { hostname = new URL(url).hostname; } catch (e) { return; }

  function isTrustedLocal(h) {
    if (h.endsWith('.gov') || h.endsWith('.edu') || h.endsWith('.mil')) return true;
    for (const d of TRUSTED_DOMAINS) {
      if (h === d || h.endsWith('.' + d)) return true;
    }
    return false;
  }

  if (isTrustedLocal(hostname)) return;

  // ── Local pre-check: should we freeze? ───────────────────────────────────
  // Only freeze on combinations with very low false-positive risk.
  const RISKY_TLDS = new Set([
    'shop', 'xyz', 'top', 'club', 'online', 'site', 'store', 'app',
    'tech', 'live', 'vip', 'pro', 'cfd', 'sbs', 'cyou', 'icu', 'buzz',
  ]);

  const SENSITIVE_PATHS = [
    /\/fpx\//i, /\/payment\//i, /\/banking\//i,
    /\/secure\//i, /\/verify\//i, /\/signin\//i,
  ];

  function localShouldFreeze(urlStr, host) {
    try {
      const u   = new URL(urlStr);
      const tld = host.split('.').pop().toLowerCase();
      const path = u.pathname;
      // IP address as hostname
      if (/^\d{1,3}(\.\d{1,3}){3}$/.test(host)) return true;
      // Credentials embedded in URL
      if (urlStr.includes('@')) return true;
      // Risky TLD + payment/banking path combo
      if (RISKY_TLDS.has(tld) && SENSITIVE_PATHS.some(p => p.test(path))) return true;
    } catch (e) {}
    return false;
  }

  const doFreeze = localShouldFreeze(url, hostname);

  // ── Apply freeze ──────────────────────────────────────────────────────────
  let freezeStyle = null;
  let safetyTimer = null;

  function applyFreeze() {
    freezeStyle = document.createElement('style');
    freezeStyle.id = '__phishing_shield_freeze__';
    freezeStyle.textContent = 'html body { display: none !important; }';
    document.documentElement.appendChild(freezeStyle);
    // Always unfreeze after 800 ms — covers Railway cold-start latency
    safetyTimer = setTimeout(removeFreeze, 800);
  }

  function removeFreeze() {
    clearTimeout(safetyTimer);
    if (freezeStyle) { freezeStyle.remove(); freezeStyle = null; }
  }

  if (doFreeze) applyFreeze();

  // ── Send to background for URL-only analysis ──────────────────────────────
  chrome.runtime.sendMessage(
    { type: 'early_url_capture', url, frozen: doFreeze },
    (response) => {
      if (!response) { removeFreeze(); return; }

      if (response.verdict === 'phishing') {
        removeFreeze(); // show our overlay instead
        injectEarlyBlock(response);
      } else {
        removeFreeze(); // safe or suspicious — let Phase 2 handle it
      }
    }
  );

  // ── Minimal block overlay (only used for hard-block cases) ────────────────
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g,
      c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  function injectEarlyBlock(data) {
    if (document.getElementById('__phishing_shield_host__')) return;

    const reason = data.reason === 'safe_browsing'
      ? 'This URL is flagged by Google Safe Browsing as a known phishing site.'
      : 'This URL matches high-confidence phishing patterns.';

    const host = document.createElement('div');
    host.id = '__phishing_shield_host__';
    host.style.cssText =
      'position:fixed;top:0;left:0;width:100%;height:100%;z-index:2147483647;pointer-events:all';
    document.documentElement.appendChild(host);

    const shadow = host.attachShadow({ mode: 'closed' });
    shadow.innerHTML = `
      <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        :host { all: initial; }
        .overlay {
          position: fixed; inset: 0;
          background: rgba(10,0,0,.96);
          display: flex; align-items: center; justify-content: center;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        .card {
          background: #180000; border: 2px solid #c0392b; border-radius: 14px;
          padding: 32px 28px; max-width: 480px; width: 92%; color: #f0f0f0;
          box-shadow: 0 8px 48px rgba(192,57,43,.4);
        }
        .icon { font-size: 38px; margin-bottom: 12px; }
        h1 { font-size: 20px; font-weight: 700; color: #e74c3c; margin-bottom: 8px; }
        .badge {
          display: inline-block; background: #2d0000; border: 1px solid #c0392b;
          border-radius: 4px; padding: 2px 8px; font-size: 11px; color: #e74c3c;
          margin-bottom: 16px;
        }
        p { font-size: 13px; color: #aaa; margin-bottom: 16px; line-height: 1.5; }
        .url-pill {
          background: #2d0000; border-radius: 6px; padding: 6px 10px;
          font-size: 11px; color: #888; word-break: break-all; margin-bottom: 20px;
        }
        .buttons { display: flex; gap: 10px; }
        .btn-back {
          flex: 1; padding: 13px; background: #c0392b; color: #fff;
          border: none; border-radius: 8px; cursor: pointer;
          font-size: 15px; font-weight: 600; transition: background .15s;
        }
        .btn-back:hover { background: #e74c3c; }
        .btn-cont {
          padding: 13px 16px; background: transparent; color: #555;
          border: 1px solid #333; border-radius: 8px; cursor: pointer;
          font-size: 12px; transition: color .15s, border-color .15s;
        }
        .btn-cont:hover { color: #888; border-color: #555; }
      </style>
      <div class="overlay">
        <div class="card">
          <div class="icon">🛡️</div>
          <h1>Phishing Site Blocked</h1>
          <div class="badge">⚡ Blocked before page loaded</div>
          <p>${escapeHtml(reason)}</p>
          <div class="url-pill">${escapeHtml(url)}</div>
          <div class="buttons">
            <button class="btn-back" id="btn-back">← Go Back to Safety</button>
            <button class="btn-cont" id="btn-cont">Continue Anyway</button>
          </div>
        </div>
      </div>`;

    shadow.getElementById('btn-back').addEventListener('click', () => window.history.back());
    shadow.getElementById('btn-cont').addEventListener('click', () => host.remove());
  }
})();
