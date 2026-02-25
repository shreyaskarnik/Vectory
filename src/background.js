// Background Service Worker
// Supports two backends:
//   - "embeddinggemma" (default): Transformers.js + EmbeddingGemma-300M in the service worker
//   - "use": MediaPipe Universal Sentence Encoder via offscreen document

import { AutoTokenizer, AutoModel, env } from "@huggingface/transformers";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const OFFSCREEN_DOCUMENT_PATH = 'offscreen.html';

/** ONNX-quantized EmbeddingGemma model */
const GEMMA_MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX";

/** EmbeddingGemma task prompt prefix */
const EMBED_TASK_PREFIX = "task: classification | query: ";

/** USE model download URL */
const USE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/1/universal_sentence_encoder.tflite';
const USE_MODEL_FILE = 'model.tflite';

/** Backend identifiers */
const BACKEND = {
    EMBEDDING_GEMMA: 'embeddinggemma',
    USE: 'use',
};

// ---------------------------------------------------------------------------
// Transformers.js config
// ---------------------------------------------------------------------------

env.allowLocalModels = false;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let creating; // offscreen document concurrency guard
let activeBackend = BACKEND.EMBEDDING_GEMMA;

// EmbeddingGemma state
let tokenizer = null;
let model = null;
let gemmaReady = false;
let gemmaLoadingPromise = null;

// ---------------------------------------------------------------------------
// Backend selection (persisted in chrome.storage.local)
// ---------------------------------------------------------------------------

async function getBackend() {
    try {
        const result = await chrome.storage.local.get('model_backend');
        return result.model_backend || BACKEND.EMBEDDING_GEMMA;
    } catch {
        return BACKEND.EMBEDDING_GEMMA;
    }
}

getBackend().then(b => { activeBackend = b; });

chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'local' && changes.model_backend) {
        activeBackend = changes.model_backend.newValue || BACKEND.EMBEDDING_GEMMA;
        if (activeBackend !== BACKEND.EMBEDDING_GEMMA) {
            tokenizer = null;
            model = null;
            gemmaReady = false;
            gemmaLoadingPromise = null;
        }
    }
});

// ---------------------------------------------------------------------------
// Offscreen document management (USE backend only)
// ---------------------------------------------------------------------------

async function setupOffscreenDocument(path) {
    const offscreenUrl = chrome.runtime.getURL(path);
    const existingContexts = await chrome.runtime.getContexts({
        contextTypes: ['OFFSCREEN_DOCUMENT'],
        documentUrls: [offscreenUrl]
    });

    if (existingContexts.length > 0) return;

    if (creating) {
        await creating;
    } else {
        creating = chrome.offscreen.createDocument({
            url: path,
            reasons: ['WORKERS'],
            justification: 'MediaPipe WASM inference engine (USE)',
        });
        await creating;
        creating = null;
    }
}

// ---------------------------------------------------------------------------
// Download progress broadcast
// ---------------------------------------------------------------------------

function broadcastProgress(sender, progress, status) {
    if (sender?.tab?.id) {
        chrome.tabs.sendMessage(sender.tab.id, {
            type: 'VECTORY_DOWNLOAD_PROGRESS',
            progress,
            status
        }).catch(() => {});
    }
}

// ---------------------------------------------------------------------------
// EmbeddingGemma: model loading
// ---------------------------------------------------------------------------

async function loadGemmaModel(sender) {
    if (gemmaReady) return;
    if (gemmaLoadingPromise) return gemmaLoadingPromise;

    gemmaLoadingPromise = (async () => {
        try {
            const hasWebGPU = !!navigator.gpu;
            const device = hasWebGPU ? "webgpu" : "wasm";
            const dtype = "q4";
            const modelSuffix = hasWebGPU ? "model_no_gather" : "model";

            console.log(`[vectory] Loading EmbeddingGemma on ${device} (dtype=${dtype})...`);
            broadcastProgress(sender, 0, 'starting');

            const progressCallback = (progress) => {
                if (progress.status === "progress") {
                    broadcastProgress(sender, (progress.progress ?? 0) / 100, 'downloading');
                }
            };

            const [loadedTokenizer, loadedModel] = await Promise.all([
                AutoTokenizer.from_pretrained(GEMMA_MODEL_ID, {
                    progress_callback: progressCallback,
                }),
                AutoModel.from_pretrained(GEMMA_MODEL_ID, {
                    device,
                    dtype,
                    model_file_name: modelSuffix,
                    progress_callback: progressCallback,
                }),
            ]);

            tokenizer = loadedTokenizer;
            model = loadedModel;
            gemmaReady = true;

            console.log(`[vectory] EmbeddingGemma ready (${device})`);
            broadcastProgress(sender, 1, 'completed');
        } catch (err) {
            console.error('[vectory] EmbeddingGemma load error:', err.message);
            broadcastProgress(sender, 0, 'failed');
            gemmaLoadingPromise = null;
            throw err;
        }
    })();

    return gemmaLoadingPromise;
}

// ---------------------------------------------------------------------------
// EmbeddingGemma: inference
// ---------------------------------------------------------------------------

async function gemmaEmbed(texts) {
    if (!tokenizer || !model) throw new Error("EmbeddingGemma not loaded");

    const prefixed = texts.map(t => EMBED_TASK_PREFIX + t);
    const inputs = tokenizer(prefixed, {
        padding: true,
        truncation: true,
        max_length: 256,
    });

    const output = await model(inputs);
    const sentenceEmbedding = output.sentence_embedding;
    const embDim = sentenceEmbedding.dims[1];
    const rawData = sentenceEmbedding.data;

    const embeddings = [];
    for (let i = 0; i < texts.length; i++) {
        const start = i * embDim;
        embeddings.push(Array.from(rawData.slice(start, start + embDim)));
    }
    return embeddings;
}

// ---------------------------------------------------------------------------
// USE backend: model download (to OPFS)
// ---------------------------------------------------------------------------

async function downloadUSEModel(sender) {
    console.log('[vectory] Starting USE model download...');
    broadcastProgress(sender, 0, 'starting');

    try {
        const response = await fetch(USE_MODEL_URL);
        if (!response.body) throw new Error('Response body is empty');

        const contentLength = response.headers.get('Content-Length');
        const total = contentLength ? parseInt(contentLength, 10) : 0;
        let loaded = 0;

        const root = await navigator.storage.getDirectory();
        const fileHandle = await root.getFileHandle(USE_MODEL_FILE, { create: true });
        const writable = await fileHandle.createWritable();
        const reader = response.body.getReader();

        await new Promise(async (resolve, reject) => {
            try {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    await writable.write(value);
                    loaded += value.length;
                    if (total > 0) broadcastProgress(sender, loaded / total, 'downloading');
                }
                await writable.close();
                broadcastProgress(sender, 1, 'completed');
                resolve();
            } catch (e) {
                try { await writable.close(); } catch (_) {}
                reject(e);
            }
        });

        console.log('[vectory] USE model download complete.');
    } catch (e) {
        console.error('[vectory] Download failed:', e);
        broadcastProgress(sender, 0, 'failed');
        throw e;
    }
}

// ---------------------------------------------------------------------------
// Message handler
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (!message.type || !message.type.startsWith('VECTORY_')) return;

    const backend = activeBackend;

    // ----------- DOWNLOAD -----------
    if (message.type === 'VECTORY_DOWNLOAD') {
        if (backend === BACKEND.EMBEDDING_GEMMA) {
            loadGemmaModel(sender)
                .then(() => sendResponse({ success: true }))
                .catch(err => sendResponse({ success: false, error: err.message }));
        } else {
            downloadUSEModel(sender)
                .then(() => sendResponse({ success: true }))
                .catch(err => sendResponse({ success: false, error: err.message }));
        }
        return true;
    }

    // ----------- EmbeddingGemma path (handled directly in background) -----------
    if (backend === BACKEND.EMBEDDING_GEMMA) {
        if (message.type === 'VECTORY_AVAILABILITY') {
            const modality = message.options?.modality || 'text';
            if (modality !== 'text') {
                sendResponse('unavailable');
                return;
            }
            if (gemmaReady) {
                sendResponse('available');
            } else if (gemmaLoadingPromise) {
                sendResponse('downloading');
            } else {
                sendResponse('downloadable');
            }
            return;
        }

        if (message.type === 'VECTORY_CREATE') {
            if (gemmaReady) {
                sendResponse({ success: true });
                return;
            }
            loadGemmaModel(sender);
            (gemmaLoadingPromise || Promise.resolve())
                .then(() => sendResponse({ success: true }))
                .catch(err => sendResponse({ success: false, error: err.message }));
            return true;
        }

        if (message.type === 'VECTORY_EMBED') {
            if (!gemmaReady) {
                sendResponse({ error: 'EmbeddingGemma not initialized' });
                return;
            }
            if (message.task === 'embed' && typeof message.text === 'string') {
                gemmaEmbed([message.text])
                    .then(([emb]) => sendResponse(emb))
                    .catch(err => sendResponse({ error: err.message }));
                return true;
            }
            if (message.task === 'embedBatch' && Array.isArray(message.texts)) {
                gemmaEmbed(message.texts)
                    .then(embs => sendResponse(embs))
                    .catch(err => sendResponse({ error: err.message }));
                return true;
            }
            sendResponse({ error: 'Invalid embed payload' });
            return;
        }

        return;
    }

    // ----------- USE path (forward to offscreen document) -----------
    setupOffscreenDocument(OFFSCREEN_DOCUMENT_PATH).then(() => {
        chrome.runtime.sendMessage(message, (response) => {
            sendResponse(response);
        });
    });
    return true;
});

// ---------------------------------------------------------------------------
// Extension action — open demo playground
// ---------------------------------------------------------------------------

chrome.action.onClicked.addListener(() => {
    chrome.tabs.create({ url: chrome.runtime.getURL('demo.html') });
});

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

chrome.runtime.onStartup.addListener(() => {
    getBackend().then(b => {
        activeBackend = b;
        if (b === BACKEND.USE) {
            setupOffscreenDocument(OFFSCREEN_DOCUMENT_PATH);
        }
    });
});
