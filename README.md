# Vectory: Speculative Web Embedding API

**Vectory** is a "[prollyfill](https://kikobeats.com/polyfill-ponyfill-and-prollyfill/#prollyfill)" that injects a tentative, built-in AI style API for embeddings into the browser.

Instead of waiting for a native `window.Embedder` implementation, Vectory provides on-device embeddings today, mimicking the architectural patterns of [built-in AI APIs](https://developer.chrome.com/docs/ai/built-in-apis) (like the Summarizer and Prompt APIs).

## Key Features

* **Native-Style API:** Follows built-in AI's `availability()` and `create()` async patterns.
* **EmbeddingGemma (default):** Google's state-of-the-art 308M parameter embedding model via [Transformers.js](https://huggingface.co/docs/transformers.js) + WebGPU/WASM.
* **Universal Sentence Encoder (alternate):** MediaPipe-based fallback via offscreen document.
* **LiteRT EmbeddingGemma:** Quantized EmbeddingGemma-300M via MediaPipe + LiteRT in offscreen document.
* **Private:** Your data never leaves your browser. All embeddings are generated locally.
* **Efficient Storage:** Models are cached automatically (HuggingFace cache for EmbeddingGemma, OPFS for USE/LiteRT).

## Known Limitations

* **No Image Embeddings:** Currently only text embeddings are supported.

## Installation & Setup

### 1. Install & Build

```bash
npm install
npm run build
```

This bundles the service worker with Transformers.js and copies extension files to `dist/`.

### 2. (Optional) MediaPipe setup for USE backend

If you want the alternate Universal Sentence Encoder backend:

```bash
npm run setup
```

### 3. Load into Chrome

1.  Open Chrome and go to `chrome://extensions`.
2.  Enable **Developer mode** in the top right corner.
3.  Click **Load unpacked**.
4.  Select the **`dist`** folder inside the project directory.

## Model Selection

Vectory defaults to **EmbeddingGemma**. To switch backends, open the extension's service worker console and run:

```javascript
// Switch to Universal Sentence Encoder
chrome.storage.local.set({ model_backend: 'use' });

// Switch to LiteRT EmbeddingGemma
chrome.storage.local.set({ model_backend: 'litertgemma' });

// Switch back to EmbeddingGemma (default)
chrome.storage.local.set({ model_backend: 'embeddinggemma' });
```

| Backend | Model | Size | Dimensions | Engine |
|---|---|---|---|---|
| `embeddinggemma` | EmbeddingGemma-300M (q4) | ~75MB | 768 | Transformers.js (WebGPU/WASM) |
| `use` | Universal Sentence Encoder | ~6MB | 512 | MediaPipe (WASM) |
| `litertgemma` | EmbeddingGemma-300M (mixed-precision) | ~150MB | 768 | MediaPipe/LiteRT (WASM) |

## Usage for Web Developers

Once the extension is installed, any website can detect and use the `Embedder` API:

```javascript
if ('Embedder' in self) {
  try {
    const status = await Embedder.availability({ modality: 'text' });

    if (status === 'available' || status === 'downloadable') {
      const embedder = await Embedder.create({
        modality: 'text',
        monitor(m) {
          m.addEventListener('downloadprogress', (e) => {
            const percent = Math.round((e.loaded / e.total) * 100);
            console.log(`Vectory: Downloading model... ${percent}%`);
          });
        }
      });

      const vector = await embedder.embed("Explore use cases for built-in embeddings.");
      console.log(`Generated ${vector.length}d vector.`);
    }
  } catch (err) {
    console.error("Failed to initialize embedder.", err);
  }
}
```

## Architecture

```
Web Page (window.Embedder)
    |  postMessage
Content Script (bridge)
    |  chrome.runtime.sendMessage
Background Service Worker
    |
    +-- EmbeddingGemma backend (default)
    |   Transformers.js runs directly in the service worker.
    |   Model: onnx-community/embeddinggemma-300m-ONNX (q4)
    |   WebGPU when available, WASM fallback.
    |
    +-- USE backend (alternate)
    |   Routes to Offscreen Document running MediaPipe WASM.
    |   Model: Universal Sentence Encoder (.tflite in OPFS)
    |
    +-- LiteRT EmbeddingGemma backend
        Routes to Offscreen Document running MediaPipe WASM.
        Model: litert-community/embeddinggemma-300m (.tflite in OPFS)
```

## Testing

```bash
npm test
```

Runs Node.js integration tests that load EmbeddingGemma via Transformers.js and verify embedding output (dimensions, cosine similarity, batch consistency).

## Roadmap

* [x] Initial MVP with Universal Sentence Encoder (~6MB).
* [x] Support for **EmbeddingGemma** via Transformers.js.
* [x] LiteRT EmbeddingGemma backend via MediaPipe.
* [ ] Multimodal support (Image Embeddings)?
* [ ] Integrated similarity utilities (`Embedder.cosineSimilarity`)?

## Contributing

Vectory is an exploratory project to accelerate product discovery for built-in web AI. We welcome feedback on use cases, the API surface, performance, etc.
