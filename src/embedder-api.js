// Embedder API (Main World)
console.log('Embedder API code loaded');

class Embedder {
    static async availability(options) {
        options = options || {};
        if (!options.modality) options.modality = 'text';
        return this._sendRequest('VECTORY_AVAILABILITY', { options });
    }

    static async create(options) {
        options = options || {};
        if (!options.modality) options.modality = 'text';

        const status = await this.availability(options);

        if (status === 'unavailable') {
            throw new Error(`Embedder is unavailable for modality: ${options.modality}`);
        }

        if (status === 'downloadable' || status === 'downloading') {
            await this._downloadModel(options.monitor);
        }

        await this._sendRequest('VECTORY_CREATE', { options });
        return new EmbedderInstance(options);
    }

    static async _downloadModel(monitorCallback) {
        const monitor = new EventTarget();
        if (monitorCallback) {
            monitorCallback(monitor);
        }

        return new Promise((resolve, reject) => {
            // Listen for progress from content script
            const progressHandler = (event) => {
                if (event.source !== window) return;
                if (event.data.type === 'VECTORY_DOWNLOAD_PROGRESS') {

                    // Construct a ProgressEvent-like object
                    // Note: We use CustomEvent because ProgressEvent constructor might not be fully flexible for 'loaded' in all contexts or we want simplicity.
                    // But Web Standards use ProgressEvent. Let's try to match it.
                    // The background sends 'progress' as 0.0 to 1.0.

                    const loaded = Math.floor(event.data.progress * 100);
                    const total = 100;

                    const progressEvent = new ProgressEvent('downloadprogress', {
                        lengthComputable: true,
                        loaded: loaded,
                        total: total
                    });

                    monitor.dispatchEvent(progressEvent);

                    if (event.data.status === 'completed') {
                        window.removeEventListener('message', progressHandler);
                    }
                    if (event.data.status === 'failed') {
                        window.removeEventListener('message', progressHandler);
                    }
                }
            };
            window.addEventListener('message', progressHandler);

            // Trigger download
            this._sendRequest('VECTORY_DOWNLOAD', {}).then(response => {
                if (response.success) {
                    resolve();
                } else {
                    reject(new Error(response.error || 'Download failed'));
                }
            });
        });
    }

    static _sendRequest(type, payload) {
        // Sanitize payload to remove non-clonable objects (like functions)
        const sanitizedPayload = JSON.parse(JSON.stringify(payload));

        return new Promise((resolve, reject) => {
            const id = Math.random().toString(36).substring(7);
            const handler = (event) => {
                if (event.source !== window) return;
                if (event.data.type === type + '_RESPONSE' && event.data.id === id) {
                    window.removeEventListener('message', handler);
                    resolve(event.data.payload);
                }
            };
            window.addEventListener('message', handler);
            window.postMessage({ type, id, ...sanitizedPayload }, '*');
        });
    }
}

class EmbedderInstance {
    constructor(options) {
        this.options = options;
    }

    async embed(text) {
        return Embedder._sendRequest('VECTORY_EMBED', { task: 'embed', text });
    }

    async embedBatch(texts) {
        return Embedder._sendRequest('VECTORY_EMBED', { task: 'embedBatch', texts });
    }

    async destroy() {
        // Todo: Implement destroy
        return;
    }
}

window.Embedder = Embedder;
console.log('Window.Embedder is set');
