import { TextEmbedder, FilesetResolver } from './lib/tasks-text.js';

console.log('Vectory Offscreen Engine Started');

let textEmbedder = null;
let modelStatus = 'unavailable'; // unavailable, downloading, downloadable, available

const MODEL_FILE_NAME = 'model.tflite';

async function checkModelInOPFS() {
    try {
        const root = await navigator.storage.getDirectory();
        // Try to get the file handle
        const fileHandle = await root.getFileHandle(MODEL_FILE_NAME);
        const file = await fileHandle.getFile();
        if (file.size > 0) {
            return { status: 'available', file };
        }
    } catch (e) {
        // File not found
    }
    return { status: 'downloadable' };
}

async function initializeEmbedder(file) {
    try {
        const vision = await FilesetResolver.forTextTasks(
            // Use local WASM files
            "./lib/wasm"
        );

        // precise url creation from blob
        const modelUrl = URL.createObjectURL(file);

        textEmbedder = await TextEmbedder.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: modelUrl
            },
            quantize: false
        });

        modelStatus = 'available';
        console.log('TextEmbedder initialized successfully');
    } catch (err) {
        console.error('Failed to initialize TextEmbedder:', err);
        modelStatus = 'unavailable';
    }
}

// Initial Check
checkModelInOPFS().then(async (result) => {
    modelStatus = result.status;
    if (modelStatus === 'available' && result.file) {
        await initializeEmbedder(result.file);
    }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    // console.log('Offscreen received message:', message);

    if (message.type === 'VECTORY_AVAILABILITY') {
        // Check OPFS again if not available, just in case
        if (modelStatus !== 'available') {
            checkModelInOPFS().then(res => {
                modelStatus = res.status;
                // If we found it now (e.g. downloaded by SW), initialize?
                // Actually SW downloads it. We should have a way to know it's ready.
                // For availability, just return status.
                sendResponse(modelStatus);
            });
            return true;
        } else {
            sendResponse('available');
        }
    }
    else if (message.type === 'VECTORY_CREATE') {
        if (modelStatus === 'available' && textEmbedder) {
            sendResponse({ success: true });
        } else {
            // Try check again?
            checkModelInOPFS().then(async res => {
                if (res.status === 'available' && res.file) {
                    await initializeEmbedder(res.file);
                    sendResponse({ success: true });
                } else {
                    sendResponse({ success: false, error: 'Model not available' });
                }
            });
            return true;
        }
    }
    else if (message.type === 'VECTORY_EMBED') {
        if (!textEmbedder) {
            sendResponse({ error: 'Embedder not initialized' });
            return;
        }

        try {
            if (message.task === 'embed') {
                const result = textEmbedder.embed(message.text);
                console.log('Embed result:', result);

                // Safety check
                if (!result.embeddings || result.embeddings.length === 0) {
                    throw new Error('No embeddings returned');
                }

                const embedding = result.embeddings[0];
                if (!embedding.floatEmbedding) {
                    const keys = Object.keys(embedding).join(', ');
                    console.error('Missing floatEmbedding. Available keys:', keys);
                    throw new Error(`No floatEmbedding in result. Keys: ${keys}`);
                }

                const float32 = embedding.floatEmbedding;
                sendResponse(Array.from(float32));
            }
            else if (message.task === 'embedBatch') {
                // MediaPipe tasks-text doesn't handle batch natively in one call usually?
                // Actually Embedder doesn't support batch in the JS API directly in the documentation usually,
                // but let's iterate.
                const results = message.texts.map(text => {
                    const res = textEmbedder.embed(text);
                    return Array.from(res.embeddings[0].floatEmbedding);
                });
                sendResponse(results);
            }
        } catch (e) {
            console.error(e);
            sendResponse({ error: e.message });
        }
    }
});
