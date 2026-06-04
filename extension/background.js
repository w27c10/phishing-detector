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
const API_URL = 'https://phishing-detector-production-dd47.up.railway.app/analyze';

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
  if (message.type !== 'analyze') return false;

  fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url: message.url, dom: message.dom }),
  })
    .then((res) => res.json())
    .then((data) => sendResponse(data))
    .catch((err) => {
      console.warn('[PhishingDetector] API error:', err.message);
      sendResponse(null);
    });

  // Return true to keep the message channel open for the async response.
  return true;
});
