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
      // Treat 'pending' as suspicious — API may not have responded yet
      // when check_redirect fires, but the URL change itself is already a signal.
      const wasSuspicious = state.verdict !== 'safe';
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
