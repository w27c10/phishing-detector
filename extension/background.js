/**
 * Background Service Worker (Manifest V3)
 *
 * Receives sanitized {url, dom} from the content script,
 * POSTs to the Flask /analyze endpoint, and returns the result.
 * Non-blocking: uses async fetch so the UI thread is never locked.
 */

// ── Update this to your Railway URL after deployment ──────────────────────────
// Local dev : 'http://localhost:5000/analyze'
// Production: 'https://your-app.up.railway.app/analyze'
const API_URL      = 'https://phishing-detector-production-dd47.up.railway.app/analyze';
const API_URL_ONLY = 'https://phishing-detector-production-dd47.up.railway.app/analyze/url';

// Per-tab state for redirect tracking.
// { tabId → { url: string, verdict: string } }
const tabState = new Map();

// ── Domain-level verdict table (session-scoped, one-way ratchet) ──────────────
// domain → { score, verdict, url, explanation_details }
// Score only ever increases — once phishing, always phishing for this session.
const domainVerdictTable = new Map();

function _extractDomain(url) {
  try {
    const parts = new URL(url).hostname.split('.');
    return parts.length >= 2 ? parts.slice(-2).join('.') : parts[0];
  } catch { return ''; }
}

// Returns true if the verdict was upgraded (new score > existing).
function _updateDomainVerdict(domain, score, verdict, url, explanationDetails) {
  const existing = domainVerdictTable.get(domain);
  if (!existing || score > existing.score) {
    domainVerdictTable.set(domain, { score, verdict, url, explanation_details: explanationDetails || {} });
    return true;
  }
  return false;
}

// Push a domain verdict upgrade to all tabs on that domain (except the sender).
function _notifyTabsForDomain(domain, payload, excludeTabId) {
  chrome.tabs.query({}, (tabs) => {
    for (const tab of tabs) {
      if (tab.id === excludeTabId || !tab.url) continue;
      if (_extractDomain(tab.url) !== domain) continue;
      chrome.tabs.sendMessage(tab.id, { type: 'domain_verdict_upgrade', ...payload }, () => {
        void chrome.runtime.lastError; // suppress "no receiver" errors
      });
    }
  });
}

// Detect SPA pushState navigation (Vue Router, React Router, etc.) and
// notify the content script to re-analyse. onHistoryStateUpdated fires for
// every history.pushState call in the page — it runs in the browser process
// so it is not affected by the content script's isolated JS world.
chrome.webNavigation.onHistoryStateUpdated.addListener((details) => {
  if (details.frameId !== 0) return; // main frame only
  chrome.tabs.sendMessage(details.tabId, {
    type: 'spa_url_changed',
    url:  details.url,
  }).catch(() => {}); // tab may not have a content script yet
});

// Inject content script into blob: pages — these are invisible to the
// manifest content_scripts declaration which only matches http/https.
// Phishing pages often open as blob: URLs to evade URL-based detection.
chrome.webNavigation.onCompleted.addListener((details) => {
  if (details.frameId !== 0) return; // main frame only
  if (!details.url.startsWith('blob:')) return;
  chrome.scripting.executeScript({
    target: { tabId: details.tabId },
    files: ['content_script.js'],
  }).catch(() => {});
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const tabId = sender.tab ? sender.tab.id : null;

  // ── Phase 1: URL-only fast check (document_start) ──────────────────────
  if (message.type === 'early_url_capture') {
    // Store URL so Phase 2 can detect if a redirect happened.
    tabState.set(tabId, { url: message.url, verdict: 'pending' });

    fetch(API_URL_ONLY, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ url: message.url }),
    })
      .then((r) => r.json())
      .then((data) => {
        tabState.set(tabId, { url: message.url, verdict: data.verdict });
        // Propagate early phishing hits to domain verdict table.
        if (data.verdict === 'phishing') {
          const domain = _extractDomain(message.url);
          if (domain) {
            const upgraded = _updateDomainVerdict(domain, data.threat_score, data.verdict, message.url, {});
            if (upgraded) _notifyTabsForDomain(domain, { threat_score: data.threat_score, verdict: data.verdict, explanation_details: {} }, tabId);
          }
        }
        sendResponse(data);
      })
      .catch((err) => {
        console.warn('[PhishingDetector] early API error:', err.message);
        tabState.set(tabId, { url: message.url, verdict: 'safe' });
        sendResponse({ verdict: 'safe', threat_score: 0 });
      });

    return true; // async
  }

  // ── Redirect check: called by Phase 2 (document_idle) ──────────────────
  // If the current URL differs from what Phase 1 captured, a redirect occurred.
  if (message.type === 'check_redirect') {
    const state = tabState.get(tabId);
    if (state && state.url && state.url !== message.currentUrl) {
      // Only treat confirmed non-safe verdicts as suspicious.
      // 'pending' means the API hasn't responded yet — not a signal on its own.
      const wasSuspicious = state.verdict === 'suspicious' || state.verdict === 'phishing';
      sendResponse({
        redirectedFrom:        wasSuspicious ? state.url     : null,
        redirectedFromVerdict: wasSuspicious ? state.verdict : null,
        earlyVerdict:          state.verdict,
      });
    } else {
      sendResponse({
        redirectedFrom:        null,
        redirectedFromVerdict: null,
        earlyVerdict:          state ? state.verdict : null,
      });
    }
    return false; // sync
  }

  // ── Domain verdict query (content script startup check) ────────────────
  if (message.type === 'get_domain_verdict') {
    const domain = _extractDomain(message.url);
    sendResponse(domain ? (domainVerdictTable.get(domain) || null) : null);
    return false;
  }

  // ── Phase 2: Full analysis (document_idle) ──────────────────────────────
  if (message.type !== 'analyze') return false;

  // If Phase 1 already hard-blocked, skip full analysis.
  // spa_navigation bypasses this: the user moved to a new SPA route and
  // needs a fresh verdict, independent of the initial page's early result.
  const early = tabState.get(tabId);
  if (early && early.verdict === 'phishing' && !message.spa_navigation) {
    sendResponse({ verdict: 'phishing', threat_score: 1.0, _skipped: true });
    return false;
  }

  // ── Domain-level session ratchet ────────────────────────────────────────
  // If this domain was already confirmed phishing in this session, skip the
  // API call and return the cached result immediately.
  const _domain = _extractDomain(message.url);
  const _domainVerdict = _domain ? domainVerdictTable.get(_domain) : null;
  if (_domainVerdict && _domainVerdict.verdict === 'phishing' && !message.spa_navigation) {
    if (tabId) tabState.set(tabId, { url: message.url, verdict: 'phishing' });
    sendResponse({
      verdict:             'phishing',
      threat_score:        _domainVerdict.score,
      explanation_details: _domainVerdict.explanation_details || {},
      _from_domain_cache:  true,
    });
    return false;
  }

  // Capture a JPEG screenshot of the tab for visual brand colour analysis,
  // then POST all signals to the backend together.
  const windowId = sender.tab ? sender.tab.windowId : chrome.windows.WINDOW_ID_CURRENT;

  chrome.tabs.captureVisibleTab(windowId, { format: 'jpeg', quality: 40 }, (dataUrl) => {
    // Strip the data URI prefix — backend only needs raw base64.
    const screenshot = dataUrl ? dataUrl.split(',')[1] : '';

    fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url:        message.url,
        dom:        message.dom,
        text:       message.text || '',
        screenshot: screenshot,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        // Update tab state with final verdict.
        if (tabId) tabState.set(tabId, { url: message.url, verdict: data.verdict });

        // Update domain-level verdict table (session ratchet — only upgrades).
        const domain = _extractDomain(message.url);
        if (domain && data.threat_score != null && data.verdict !== 'safe') {
          const upgraded = _updateDomainVerdict(
            domain, data.threat_score, data.verdict,
            message.url, data.explanation_details
          );
          if (upgraded) {
            _notifyTabsForDomain(domain, {
              threat_score:        data.threat_score,
              verdict:             data.verdict,
              explanation_details: data.explanation_details || {},
            }, tabId);
          }
        }

        sendResponse(data);
      })
      .catch((err) => {
        console.warn('[PhishingDetector] API error:', err.message);
        sendResponse(null);
      });
  });

  // Return true to keep the message channel open for the async response.
  return true;
});
