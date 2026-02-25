/**
 * Integration tests for LiteRT EmbeddingGemma backend.
 *
 * These run in Node.js and verify that:
 *   1. The .tflite model is downloadable from HuggingFace
 *   2. The downloaded file is a valid TFLite flatbuffer
 *   3. The file size is reasonable for a 300M parameter model
 *   4. The model constants in background.js match the test expectations
 *
 * NOTE: Full embedding inference tests are not possible in Node.js because
 * the LiteRT backend runs via MediaPipe WASM in a browser offscreen document.
 * EmbeddingGemma embedding quality is validated in embedding-gemma.test.js
 * (same model, Transformers.js runtime).
 *
 * Uses Node.js built-in test runner (node --test).
 * Skips gracefully when network is unavailable or model is gated (401).
 */

import { describe, it, before } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

// ---------------------------------------------------------------------------
// Config — must stay in sync with src/background.js
// ---------------------------------------------------------------------------

const LITERT_GEMMA_MODEL_URL =
  "https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/embeddinggemma-300M_seq512_mixed-precision.tflite";

const LITERT_GEMMA_MODEL_FILE = "embeddinggemma-litert.tflite";

// TFLite flatbuffer magic bytes at offset 4: "TFL3"
const TFLITE_MAGIC = new Uint8Array([0x54, 0x46, 0x4c, 0x33]);

const __dirname = dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Network check
// ---------------------------------------------------------------------------

async function checkNetwork() {
  try {
    await fetch("https://huggingface.co/api/whoami-v2", {
      method: "HEAD",
      signal: AbortSignal.timeout(5000),
    });
    return true;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("LiteRT EmbeddingGemma model", { timeout: 120_000 }, () => {
  let networkAvailable;
  let modelAccessible; // false when gated model returns 401
  let headResponse;

  before(async () => {
    networkAvailable = await checkNetwork();
    if (!networkAvailable) {
      console.log("# Skipping LiteRT EmbeddingGemma tests — no network access.");
      return;
    }

    headResponse = await fetch(LITERT_GEMMA_MODEL_URL, {
      method: "HEAD",
      signal: AbortSignal.timeout(15000),
      redirect: "follow",
    });

    modelAccessible = headResponse.status === 200;
    if (!modelAccessible) {
      console.log(
        `# Skipping LiteRT model download tests — HTTP ${headResponse.status} ` +
        `(model is gated and requires HuggingFace license acceptance).`
      );
    }
  });

  it("model URL is reachable (200 or 401 gated)", async (t) => {
    if (!networkAvailable) return t.skip("no network");
    assert.ok(
      headResponse.status === 200 || headResponse.status === 401,
      `Expected 200 or 401, got ${headResponse.status}`
    );
  });

  it("model file is larger than 50 MB", async (t) => {
    if (!networkAvailable) return t.skip("no network");
    if (!modelAccessible) return t.skip("model gated (401)");
    const contentLength = headResponse.headers.get("content-length");
    assert.ok(contentLength, "Content-Length header should be present");
    const size = parseInt(contentLength, 10);
    const fiftyMB = 50 * 1024 * 1024;
    assert.ok(size > fiftyMB, `Expected > 50 MB, got ${(size / 1024 / 1024).toFixed(1)} MB`);
  });

  it("content-type indicates a binary file", async (t) => {
    if (!networkAvailable) return t.skip("no network");
    if (!modelAccessible) return t.skip("model gated (401)");
    const ct = headResponse.headers.get("content-type") || "";
    assert.ok(
      ct.includes("octet-stream") || ct.includes("flatbuffers") || ct.includes("application/"),
      `Expected binary content-type, got "${ct}"`
    );
  });

  it("first bytes match TFLite flatbuffer magic", async (t) => {
    if (!networkAvailable) return t.skip("no network");
    if (!modelAccessible) return t.skip("model gated (401)");

    // Download just the first 8 bytes via Range header
    const rangeRes = await fetch(LITERT_GEMMA_MODEL_URL, {
      headers: { Range: "bytes=0-7" },
      signal: AbortSignal.timeout(15000),
      redirect: "follow",
    });

    const buf = new Uint8Array(await rangeRes.arrayBuffer());
    // TFLite magic is at bytes 4-7
    const magic = buf.slice(4, 8);

    assert.deepEqual(
      Array.from(magic),
      Array.from(TFLITE_MAGIC),
      `Expected TFLite magic bytes "TFL3" at offset 4, got [${Array.from(magic).map(b => "0x" + b.toString(16).padStart(2, "0")).join(", ")}]`
    );
  });

  it("background.js model URL matches test URL", async (t) => {
    const bgPath = resolve(__dirname, "..", "src", "background.js");
    const bgSource = readFileSync(bgPath, "utf8");
    assert.ok(
      bgSource.includes(LITERT_GEMMA_MODEL_URL),
      "background.js should contain the LiteRT model URL"
    );
    assert.ok(
      bgSource.includes(`'${LITERT_GEMMA_MODEL_FILE}'`) || bgSource.includes(`"${LITERT_GEMMA_MODEL_FILE}"`),
      "background.js should contain the OPFS file name"
    );
  });
});
