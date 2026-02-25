import { TextEmbedder, FilesetResolver } from './lib/tasks-text.js';

console.log('Vectory Offscreen Engine Started');

let textEmbedder = null;
let modelStatus = 'unavailable'; // unavailable, downloading, downloadable, available
let activeModelFile = null; // tracks which model file is currently loaded

/** OPFS file names per backend */
const MODEL_FILES = {
    use: 'model.tflite',
    litertgemma: 'embeddinggemma-litert.tflite',
};

function getModelFileName(backend) {
    return MODEL_FILES[backend] || MODEL_FILES.use;
}

async function checkModelInOPFS(fileName) {
    try {
        const root = await navigator.storage.getDirectory();
        const fileHandle = await root.getFileHandle(fileName);
        const file = await fileHandle.getFile();
        if (file.size > 0) {
            return { status: 'available', file };
        }
    } catch (e) {
        // File not found
    }
    return { status: 'downloadable' };
}

async function initializeEmbedder(file, fileName) {
    try {
        const vision = await FilesetResolver.forTextTasks(
            "./lib/wasm"
        );

        const modelUrl = URL.createObjectURL(file);

        textEmbedder = await TextEmbedder.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: modelUrl
            },
            quantize: false
        });

        activeModelFile = fileName;
        modelStatus = 'available';
        console.log(`TextEmbedder initialized successfully (${fileName})`);
    } catch (err) {
        console.error(`Failed to initialize TextEmbedder (${fileName}):`, err);
        modelStatus = 'unavailable';
    }
}

// Initial Check — try USE model by default
checkModelInOPFS(MODEL_FILES.use).then(async (result) => {
    modelStatus = result.status;
    if (modelStatus === 'available' && result.file) {
        await initializeEmbedder(result.file, MODEL_FILES.use);
    }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    const backend = message._backend || 'use';
    const modelFile = getModelFileName(backend);

    if (message.type === 'VECTORY_AVAILABILITY') {
        const modality = message.options?.modality || 'text';
        if (modality !== 'text') {
            sendResponse('unavailable');
            return;
        }

        // If we have the right model loaded, report available
        if (modelStatus === 'available' && activeModelFile === modelFile) {
            sendResponse('available');
            return;
        }

        // Check OPFS for the requested model
        checkModelInOPFS(modelFile).then(res => {
            modelStatus = res.status;
            sendResponse(modelStatus);
        });
        return true;
    }
    else if (message.type === 'VECTORY_CREATE') {
        const modality = message.options?.modality || 'text';
        if (modality !== 'text') {
            sendResponse({ success: false, error: 'Unsupported modality' });
            return;
        }

        // If the right model is already loaded, we're done
        if (modelStatus === 'available' && textEmbedder && activeModelFile === modelFile) {
            sendResponse({ success: true });
            return;
        }

        // Load the requested model from OPFS
        checkModelInOPFS(modelFile).then(async res => {
            if (res.status === 'available' && res.file) {
                await initializeEmbedder(res.file, modelFile);
                sendResponse({ success: true });
            } else {
                sendResponse({ success: false, error: 'Model not available' });
            }
        });
        return true;
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
