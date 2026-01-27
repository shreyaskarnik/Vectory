// Background Service Worker
const OFFSCREEN_DOCUMENT_PATH = 'offscreen.html';
let creating; // A global promise to avoid concurrency issues

async function setupOffscreenDocument(path) {
    // Check all existing offscreen documents.
    const offscreenUrl = chrome.runtime.getURL(path);
    const existingContexts = await chrome.runtime.getContexts({
        contextTypes: ['OFFSCREEN_DOCUMENT'],
        documentUrls: [offscreenUrl]
    });

    if (existingContexts.length > 0) {
        return;
    }

    // Create document.
    if (creating) {
        await creating;
    } else {
        creating = chrome.offscreen.createDocument({
            url: path,
            reasons: ['WORKERS'],
            justification: 'Inference Engine for embeddings',
        });
        await creating;
        creating = null;
    }
}

chrome.runtime.onStartup.addListener(() => {
    setupOffscreenDocument(OFFSCREEN_DOCUMENT_PATH);
});

// Also ensure it's created on message if not exists
// Also ensure it's created on message if not exists
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'VECTORY_DOWNLOAD') {
        downloadModel(sender).then(() => {
            sendResponse({ success: true });
        }).catch(err => {
            sendResponse({ success: false, error: err.message });
        });
        return true;
    }

    if (message.type && message.type.startsWith('VECTORY_')) {
        setupOffscreenDocument(OFFSCREEN_DOCUMENT_PATH).then(() => {
            // Forward to Offscreen
            chrome.runtime.sendMessage(message, (response) => {
                sendResponse(response);
            });
        });
        return true; // Keep channel open
    }
});

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/1/universal_sentence_encoder.tflite';
const MODEL_FILE_NAME = 'model.tflite';

async function downloadModel(sender) {
    console.log('Starting model download...');
    broadcastProgress(sender, 0, 'starting');

    try {
        const response = await fetch(MODEL_URL);
        if (!response.body) throw new Error('Response body is empty');

        const contentLength = response.headers.get('Content-Length');
        const total = contentLength ? parseInt(contentLength, 10) : 0;
        let loaded = 0;

        const root = await navigator.storage.getDirectory();

        // delete existing if any, to be clean?
        // await root.removeEntry(MODEL_FILE_NAME).catch(() => {});

        const fileHandle = await root.getFileHandle(MODEL_FILE_NAME, { create: true });
        const writable = await fileHandle.createWritable();

        const reader = response.body.getReader();

        await new Promise(async (resolve, reject) => {
            try {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    await writable.write(value);
                    loaded += value.length;

                    if (total > 0) {
                        broadcastProgress(sender, loaded / total, 'downloading');
                    }
                }
                await writable.close();
                broadcastProgress(sender, 1, 'completed');
                resolve();
            } catch (e) {
                // Try to close
                try { await writable.close(); } catch (_) { }
                reject(e);
            }
        });

        console.log('Model download complete.');

    } catch (e) {
        console.error('Download failed:', e);
        broadcastProgress(sender, 0, 'failed');
        throw e;
    }
}

function broadcastProgress(sender, progress, status) {
    // Send back to the specific tab that requested it
    if (sender.tab && sender.tab.id) {
        chrome.tabs.sendMessage(sender.tab.id, {
            type: 'VECTORY_DOWNLOAD_PROGRESS',
            progress,
            status
        });
    }
}
