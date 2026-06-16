/**
 * Content Script — Local Sanitizer, Scraper, and Warning UI
 *
 * Runs at document_idle inside every browser tab.
 *
 * Steps:
 *   1. Capture URL and clone + sanitize the DOM (strip all input values).
 *   2. Send sanitized payload to the background service worker.
 *   3. If threat_score >= 0.75, inject a Shadow DOM warning overlay.
 */

(function () {
  'use strict';

  // ── 1. Trusted-domain bypass (runs first, before any DOM access) ──────────
  // Well-known legitimate domains are allowlisted to avoid false positives.
  // Check is done BEFORE cloning the DOM so we never touch YouTube / Google
  // / etc. at all — avoiding any interference with their page lifecycle.

  const TRUSTED_DOMAINS = new Set([
    'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'x.com',
    'instagram.com', 'linkedin.com', 'github.com', 'apple.com', 'microsoft.com',
    'amazon.com', 'netflix.com', 'wikipedia.org', 'reddit.com', 'bing.com',
    'yahoo.com', 'whatsapp.com', 'telegram.org',
  ]);

  function isTrusted(hostname) {
    for (const d of TRUSTED_DOMAINS) {
      if (hostname === d || hostname.endsWith('.' + d)) return true;
    }
    if (hostname.endsWith('.gov') || hostname.endsWith('.edu') ||
        hostname.endsWith('.mil')) return true;
    return false;
  }

  const rawUrl = window.location.href;

  // For blob: URLs, extract the embedded HTTPS origin as the effective URL.
  let url = rawUrl;
  if (rawUrl.startsWith('blob:')) {
    try {
      url = new URL(rawUrl.slice(5)).origin;
    } catch (e) {
      url = rawUrl;
    }
  }

  const effectiveHostname = new URL(url).hostname;
  if (isTrusted(effectiveHostname)) return; // ← exit before touching DOM

  // ── 2. Scrape & sanitise ──────────────────────────────────────────────────

  // Clone the DOM so we never touch the live page.
  const domClone = document.documentElement.cloneNode(true);

  // Strip all input values to satisfy the zero-knowledge privacy requirement.
  domClone.querySelectorAll('input').forEach((el) => {
    el.value = '';
    el.removeAttribute('value');
  });

  // Cap DOM payload at 500 KB to stay within the 500 ms latency envelope.
  const MAX_DOM_BYTES = 500 * 1024;
  let dom = domClone.outerHTML;
  if (dom.length > MAX_DOM_BYTES) {
    dom = dom.slice(0, MAX_DOM_BYTES);
  }

  // ── 3. Send to background worker ─────────────────────────────────────────

  // Extract only PROMINENT text for brand impersonation detection:
  // title, headings, and submit buttons. A phishing page puts the spoofed
  // brand name in its heading. A real site mentioning another brand in its
  // footer/payment-options section should NOT trigger the brand detector.
  const prominentSelectors = 'title, h1, h2, h3, button[type="submit"], input[type="submit"]';
  const prominentText = Array.from(document.querySelectorAll(prominentSelectors))
    .map(el => el.innerText || el.value || el.textContent || '')
    .join(' ')
    .slice(0, 2000);
  const text = prominentText;

  // ── Check for redirect from suspicious site (Phase 1 → Phase 2 handoff) ──
  chrome.runtime.sendMessage({ type: 'check_redirect', currentUrl: url }, (redirectInfo) => {
    if (redirectInfo && redirectInfo.redirectedFrom) {
      injectRedirectBanner(redirectInfo.redirectedFrom, redirectInfo.redirectedFromVerdict);
    }

    // Skip full analysis if Phase 1 already hard-blocked this tab.
    if (redirectInfo && redirectInfo.earlyVerdict === 'phishing') return;

    // ── Full analysis (Phase 2) ─────────────────────────────────────────────
    console.log('[PhishingDetector] text sent to backend:', text);
    chrome.runtime.sendMessage({ type: 'analyze', url, dom, text }, (response) => {
      if (chrome.runtime.lastError) {
        console.warn('[PhishingDetector]', chrome.runtime.lastError.message);
        return;
      }
      console.log('[PhishingDetector] response:', JSON.stringify(response));
      if (response && response.verdict === 'phishing') {
        injectWarning(response);
      } else if (response && response.verdict === 'suspicious') {
        injectCautionBanner(response);
      }
    });
  });

  // ── 3. Warning overlay (Shadow DOM) ──────────────────────────────────────

  function escapeHtml(str) {
    return String(str).replace(
      /[&<>"']/g,
      (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])
    );
  }

  function riskClass(score) {
    if (score >= 0.75) return 'high';
    if (score >= 0.5) return 'med';
    return 'low';
  }

  function renderFactor(label, score, message) {
    if (!score) return '';   // hide 0% and undefined/null factors
    const pct = Math.round(score * 100);
    const cls = riskClass(score);
    return `
      <div class="factor">
        <div class="factor-left">
          <div class="factor-name">${label}</div>
          ${message ? `<div class="factor-msg">${escapeHtml(message)}</div>` : ''}
        </div>
        <div class="factor-score ${cls}">${pct}%</div>
      </div>`;
  }

  // Shown on the destination page after a redirect from a suspicious/phishing URL.
  function injectRedirectBanner(fromUrl, fromVerdict) {
    if (document.getElementById('__phishing_shield_redirect_host__')) return;

    const isPhishing = fromVerdict === 'phishing';
    const color      = isPhishing ? '#7a0000' : '#7a4f00';
    const border     = isPhishing ? '#e74c3c' : '#f0a500';
    const textColor  = isPhishing ? '#ffb3b3' : '#ffe8a3';
    const titleColor = isPhishing ? '#ff6b6b' : '#ffd166';
    const label      = isPhishing ? '🚨 Redirected from phishing site' : '⚠️ Redirected from suspicious site';

    const host = document.createElement('div');
    host.id = '__phishing_shield_redirect_host__';
    host.style.cssText =
      'position:fixed;top:0;left:0;width:100%;z-index:2147483647;pointer-events:none';
    document.documentElement.appendChild(host);

    const shadow = host.attachShadow({ mode: 'closed' });
    shadow.innerHTML = `
      <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        :host { all: initial; }
        .banner {
          width: 100%; background: ${color}; border-bottom: 2px solid ${border};
          padding: 10px 16px; display: flex; align-items: center; gap: 12px;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          font-size: 13px; color: ${textColor};
        }
        .body { flex: 1; line-height: 1.5; }
        .title { font-weight: 700; color: ${titleColor}; }
        .origin {
          display: inline-block; background: rgba(0,0,0,.25); border-radius: 4px;
          padding: 1px 6px; font-size: 11px; font-family: monospace; word-break: break-all;
        }
        .banner { pointer-events: auto; }
        .btn-close {
          background: transparent; border: 1px solid rgba(255,255,255,.25);
          border-radius: 6px; color: ${titleColor}; cursor: pointer;
          font-size: 13px; padding: 4px 10px; flex-shrink: 0;
          transition: background .15s;
        }
        .btn-close:hover { background: rgba(255,255,255,.1); }
      </style>
      <div class="banner">
        <div class="body">
          <span class="title">${label}</span><br>
          Origin: <span class="origin">${escapeHtml(fromUrl)}</span>
        </div>
        <button class="btn-close" id="btn-close">Dismiss ✕</button>
      </div>`;

    shadow.getElementById('btn-close').addEventListener('click', () => host.remove());
  }

  // Returns the top (up to 3) factor labels with score ≥ 0.5, sorted descending.
  function topFactors(exp) {
    return [
      ['URL Analysis',        exp.url_threat_factor],
      ['Page Structure',      exp.dom_threat_factor],
      ['Metadata Behaviour',  exp.metadata_threat_factor],
      ['Brand Impersonation', exp.brand_threat_factor],
      ['Domain Age',          exp.domain_age_factor],
      ['Link Clustering',     exp.link_cluster_factor],
      ['Dead Links',          exp.dead_link_factor],
      ['Broken Links',        exp.broken_link_factor],
      ['Payment Form',        exp.payment_form_factor],
    ]
      .filter(([, s]) => typeof s === 'number' && s >= 0.5)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([label]) => label);
  }

  function injectCautionBanner(data) {
    if (document.getElementById('__phishing_shield_banner_host__')) return;

    const exp      = data.explanation_details || {};
    const totalPct = Math.round(data.threat_score * 100);
    const reasons  = topFactors(exp);
    const pillsHtml = reasons.length
      ? reasons.map(r => `<span class="pill">${escapeHtml(r)}</span>`).join('')
      : '<span class="pill">Elevated risk score</span>';

    const host = document.createElement('div');
    host.id = '__phishing_shield_banner_host__';
    host.style.cssText =
      'position:fixed;top:0;left:0;width:100%;z-index:2147483647;pointer-events:none';
    document.documentElement.appendChild(host);

    const shadow = host.attachShadow({ mode: 'closed' });

    shadow.innerHTML = `
      <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        :host { all: initial; }

        .banner {
          width: 100%;
          background: #7a4f00;
          border-bottom: 2px solid #f0a500;
          padding: 10px 16px;
          display: flex;
          align-items: center;
          gap: 12px;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          font-size: 13px;
          color: #ffe8a3;
        }
        .icon { font-size: 18px; flex-shrink: 0; }
        .body { flex: 1; line-height: 1.5; }
        .title { font-weight: 700; color: #ffd166; }
        .score { font-weight: 600; color: #ffd166; }
        .pills { display: inline-flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
        .pill {
          background: rgba(255,209,102,0.15);
          border: 1px solid rgba(255,209,102,0.4);
          border-radius: 20px;
          padding: 2px 8px;
          font-size: 11px;
          color: #ffd166;
        }
        .btn-close {
          background: transparent;
          border: 1px solid rgba(255,209,102,0.4);
          border-radius: 6px;
          color: #ffd166;
          cursor: pointer;
          font-size: 13px;
          padding: 4px 10px;
          flex-shrink: 0;
          transition: background .15s;
        }
        .btn-close:hover { background: rgba(255,209,102,0.15); }
        .banner { pointer-events: auto; }
      </style>

      <div class="banner">
        <span class="icon">⚠️</span>
        <div class="body">
          <span class="title">Proceed with caution</span>
          — threat score <span class="score">${totalPct}%</span>.
          Suspicious signals detected:
          <div class="pills">${pillsHtml}</div>
        </div>
        <button class="btn-close" id="btn-close">Dismiss ✕</button>
      </div>
    `;

    shadow.getElementById('btn-close').addEventListener('click', () => host.remove());
  }

  function injectWarning(data) {
    // Prevent double-injection.
    if (document.getElementById('__phishing_shield_host__')) return;

    const exp = data.explanation_details || {};
    const totalPct = Math.round(data.threat_score * 100);

    // Host element — positioned over the entire viewport.
    const host = document.createElement('div');
    host.id = '__phishing_shield_host__';
    host.style.cssText =
      'position:fixed;top:0;left:0;width:100%;height:100%;z-index:2147483647;pointer-events:all';
    document.documentElement.appendChild(host);

    // Closed shadow root prevents the host page from tampering with our UI.
    const shadow = host.attachShadow({ mode: 'closed' });

    shadow.innerHTML = `
      <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        :host { all: initial; }

        .overlay {
          position: fixed; inset: 0;
          background: rgba(10, 0, 0, 0.96);
          display: flex; align-items: center; justify-content: center;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        .card {
          background: #180000;
          border: 2px solid #c0392b;
          border-radius: 14px;
          padding: 32px 28px;
          max-width: 540px;
          width: 92%;
          color: #f0f0f0;
          box-shadow: 0 8px 48px rgba(192,57,43,0.4);
        }

        .header {
          display: flex;
          align-items: flex-start;
          gap: 14px;
          margin-bottom: 16px;
        }
        .shield-icon { font-size: 38px; line-height: 1; flex-shrink: 0; }
        .header h1 { font-size: 20px; font-weight: 700; color: #e74c3c; margin-bottom: 4px; }
        .header p  { font-size: 13px; color: #aaa; }

        .url-pill {
          background: #2d0000;
          border-radius: 6px;
          padding: 6px 10px;
          font-size: 11px;
          color: #888;
          word-break: break-all;
          margin-bottom: 18px;
        }

        .score-bar {
          background: #2d0000;
          border-radius: 10px;
          padding: 14px 16px;
          margin-bottom: 18px;
          display: flex;
          align-items: baseline;
          gap: 8px;
        }
        .score-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: .06em; }
        .score-value { font-size: 32px; font-weight: 800; color: #e74c3c; }

        .factors { margin-bottom: 24px; }
        .factor {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          padding: 10px 0;
          border-bottom: 1px solid #2d0000;
          gap: 12px;
        }
        .factor:last-child { border-bottom: none; }
        .factor-left { flex: 1; }
        .factor-name { font-size: 13px; font-weight: 600; color: #ccc; }
        .factor-msg  { font-size: 11px; color: #777; margin-top: 3px; line-height: 1.4; }
        .factor-score { font-size: 14px; font-weight: 700; flex-shrink: 0; }
        .high { color: #e74c3c; }
        .med  { color: #e67e22; }
        .low  { color: #f1c40f; }

        .buttons { display: flex; gap: 10px; }
        .btn-back {
          flex: 1;
          padding: 13px;
          background: #c0392b;
          color: #fff;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          font-size: 15px;
          font-weight: 600;
          transition: background .15s;
        }
        .btn-back:hover { background: #e74c3c; }
        .btn-continue {
          padding: 13px 16px;
          background: transparent;
          color: #555;
          border: 1px solid #333;
          border-radius: 8px;
          cursor: pointer;
          font-size: 12px;
          transition: color .15s, border-color .15s;
        }
        .btn-continue:hover { color: #888; border-color: #555; }
      </style>

      <div class="overlay">
        <div class="card">
          <div class="header">
            <span class="shield-icon">🛡️</span>
            <div>
              <h1>Phishing Site Detected</h1>
              <p>This page has been identified as a potential threat. Do not enter any credentials.</p>
            </div>
          </div>

          <div class="url-pill">${escapeHtml(url)}</div>

          <div class="score-bar">
            <span class="score-label">Threat Score</span>
            <span class="score-value">${totalPct}%</span>
          </div>

          <div class="factors">
            ${renderFactor('URL Analysis',        exp.url_threat_factor,      exp.url_diagnostic_message)}
            ${renderFactor('Page Structure',      exp.dom_threat_factor,      exp.dom_diagnostic_message)}
            ${renderFactor('Metadata Behaviour',  exp.metadata_threat_factor, exp.metadata_diagnostic_message)}
            ${renderFactor('Brand Impersonation', exp.brand_threat_factor,    exp.brand_diagnostic_message)}
            ${renderFactor('Domain Age',          exp.domain_age_factor,      exp.domain_age_message)}
            ${renderFactor('Broken Links',        exp.broken_link_factor,     exp.broken_link_message)}
            ${renderFactor('Payment Form',        exp.payment_form_factor,    exp.payment_form_message)}
            ${renderFactor('Visual Analysis',     exp.visual_threat_factor,   exp.visual_diagnostic_message)}
          </div>

          <div class="buttons">
            <button class="btn-back" id="btn-back">← Go Back to Safety</button>
            <button class="btn-continue" id="btn-continue">Continue Anyway</button>
          </div>
        </div>
      </div>
    `;

    shadow.getElementById('btn-back').addEventListener('click', () => {
      window.history.back();
    });

    shadow.getElementById('btn-continue').addEventListener('click', () => {
      host.remove();
    });
  }

  // ── SPA navigation detection ──────────────────────────────────────────────
  // Hash and pushState navigations don't reload the document, so content
  // scripts don't re-run. We intercept URL changes and re-trigger analysis.

  let _spaLastUrl = location.href;
  let _spaTimer   = null;

  function _removeSpaBanners() {
    ['__phishing_shield_host__', '__phishing_shield_banner_host__']
      .forEach(id => { const el = document.getElementById(id); if (el) el.remove(); });
  }

  function _runSpaAnalysis() {
    const currentUrl = location.href;

    const domCloneSpa = document.documentElement.cloneNode(true);
    domCloneSpa.querySelectorAll('input').forEach(el => {
      el.value = ''; el.removeAttribute('value');
    });
    let domSpa = domCloneSpa.outerHTML;
    if (domSpa.length > 500 * 1024) domSpa = domSpa.slice(0, 500 * 1024);

    const textSpa = Array.from(document.querySelectorAll(
      'title, h1, h2, h3, button[type="submit"], input[type="submit"]'
    )).map(el => el.innerText || el.value || el.textContent || '').join(' ').slice(0, 2000);

    chrome.runtime.sendMessage(
      { type: 'analyze', url: currentUrl, dom: domSpa, text: textSpa, spa_navigation: true },
      (response) => {
        if (chrome.runtime.lastError || !response) return;
        console.log('[PhishingDetector] SPA re-analysis:', currentUrl, JSON.stringify(response));
        if (response.verdict === 'phishing') injectWarning(response);
        else if (response.verdict === 'suspicious') injectCautionBanner(response);
      }
    );
  }

  function _onSpaNavigation() {
    const newUrl = location.href;
    if (newUrl === _spaLastUrl) return;
    _spaLastUrl = newUrl;

    clearTimeout(_spaTimer);
    _spaTimer = setTimeout(() => {
      _removeSpaBanners();
      _runSpaAnalysis();
    }, 1200);
  }

  window.addEventListener('hashchange', _onSpaNavigation);
  window.addEventListener('popstate',   _onSpaNavigation);

  // pushState / replaceState: intercepted via background's
  // onHistoryStateUpdated (runs in browser process, not isolated JS world).
  chrome.runtime.onMessage.addListener((message) => {
    if (message.type === 'spa_url_changed') _onSpaNavigation();
    return false;
  });

})();
