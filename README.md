# 🚀 Vectory: Speculative Web Embedding API

**Vectory** is a "[prollyfill](https://kikobeats.com/polyfill-ponyfill-and-prollyfill/#prollyfill)" that injects a tentative, built-in AI style API for embeddings into the browser.

Instead of waiting for a native `window.Embedder` implementation, Vectory uses **Google MediaPipe** to provide on-device embeddings today, mimicking the architectural patterns of [built-in AI APIs](https://developer.chrome.com/docs/ai/built-in-apis) (like the Summarizer and Prompt APIs).

## ✨ Key Features

* **Native-Style API:** Follows built-in AI's `availability()` and `create()` async patterns.
* **On-Device Inference:** Powered by Google MediaPipe and the Universal Sentence Encoder (default for now; aiming to switch to EmbeddingGemma).
* **Private:** Your data never leaves your browser. All embeddings are generated locally.
* **Efficient Storage:** Models are downloaded and stored once for all, in the **Origin Private File System (OPFS)** for fast, direct-to-disk access.

## ✨ Known Limitations

* **No Image Embeddings:** Currently only text embeddings are supported.

## 📦 Installation & Setup

### 1. Download & Setup
Clone the repository and install the necessary dependencies (this downloads the MediaPipe assets to `src/lib`).

```
npm install
npm run setup
```

### 2. Load into Chrome
1.  Open Chrome and go to `chrome://extensions`.
2.  Enable **Developer mode** in the top right corner.
3.  Click **Load unpacked**.
4.  Select the `src` folder inside the project directory (e.g., `[...]/vectory/src`).

That's it! Usage details are below.

## 🛠️ Usage for Web Developers

Once the extension is installed, any website / extension can detect and use the `Embedder` API:

```javascript
// 0. Feature Detection: Check if the Vectory Prollyfill is present
if ('Embedder' in self) {
  try {
    // 1. Check if this device can run a particular type of embedding
    const status = await Embedder.availability({ modality: 'text' });

    if (status === 'available' || status === 'downloadable') {
      // 2. Create the session (handles download if needed)
      const embedder = await Embedder.create({ 
        modality: 'text',
        monitor(m) {
          m.addEventListener('downloadprogress', (e) => {
            // e.loaded and e.total provide the progress for that 6MB model
            const percent = Math.round((e.loaded / e.total) * 100);
            console.log(`Vectory: Downloading model... ${percent}%`);
          });
        }
      });

      // 3. Generate vectors
      const input = "Help us explore use cases and refine the API surface for a built-in Embedding capability.";
      const vector = await embedder.embed(input);
      
      console.log(`Vectory: Generated ${vector.length}d vector successfully.`);
      // console.log(vector); 
    } else {
      console.warn("Vectory: Embedding is not supported on this device.", status);
    }
  } catch (err) {
    console.error("Vectory: Failed to initialize embedder.", err);
  }
} else {
  console.log("Vectory: Embedder API not found. Please ensure the extension is installed and enabled.");
}
```

## 🏗️ Architecture

1. **Main World Injection:** Injects the `Embedder` class so it's accessible to the page's own JS.
2. **Service Worker:** Orchestrates model downloads and manages OPFS file handles.
3. **Offscreen Document:** Runs the MediaPipe WASM runtime in a dedicated environment to keep the Service Worker lean and avoid memory constraints.

## 🚀 Roadmap

* [x] Initial MVP with Universal Sentence Encoder (~6MB).
* [ ] Support for **EmbeddingGemma**.
* [ ] Multimodal support (Image Embeddings via MediaPipe)?
* [ ] Integrated similarity utilities (`Embedder.cosineSimilarity`)?

## 🤝 Contributing

Vectory is an exploratory project to accelerate product discovery for built-in web AI. We welcome feedback on use cases, the API surface, performance, etc.