// Content Script
console.log('Vectory Content Script Injected');

// Inject the Embedder API
const script = document.createElement('script');
script.src = chrome.runtime.getURL('embedder-api.js');
script.onload = () => {
    script.remove();
};
(document.head || document.documentElement).appendChild(script);

// Bridge: Listen for messages from web page (Embedder API) and forward to Background
window.addEventListener('message', (event) => {
    // We only accept messages from ourselves
    if (event.source !== window) return;

    if (event.data.type && event.data.type.startsWith('VECTORY_')) {
        // console.log('Content Script forwarding message to Background:', event.data);
        chrome.runtime.sendMessage(event.data, (response) => {
            // Send response back to page
            window.postMessage({
                type: event.data.type + '_RESPONSE',
                id: event.data.id,
                payload: response
            }, '*');
        });
    }
});

// Listen for messages from Background (e.g. download progress)
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'VECTORY_DOWNLOAD_PROGRESS') {
        window.postMessage(message, '*');
    }
});
