/**
 * Integration tests for LiteRT EmbeddingGemma backend.
 *
 * These run in Node.js and verify that:
 *   1. The .tflite model is downloadable from HuggingFace
 *   2. The downloaded file is a valid TFLite flatbuffer
 *   3. The file size is reasonable for a 300M parameter model
 *
 * Uses Node.js built-in test runner (node --test).
 * Skips gracefully when network is unavailable.
 */

import { describe, it, before } from "node:test";
import assert from "node:assert/strict";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const LITERT_GEMMA_MODEL_URL =
  "https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/embeddinggemma-300M_seq512_mixed-precision.tflite";

// TFLite flatbuffer magic bytes: "TFL3"
const TFLITE_MAGIC = new Uint8Array([0x54, 0x46, 0x4c, 0x33]);

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
  });

  it("model URL is reachable and returns 200", async (t) => {
    if (!networkAvailable) return t.skip("no network");
    assert.equal(headResponse.status, 200, `Expected 200, got ${headResponse.status}`);
  });

  it("model file is larger than 50 MB", async (t) => {
    if (!networkAvailable) return t.skip("no network");
    const contentLength = headResponse.headers.get("content-length");
    assert.ok(contentLength, "Content-Length header should be present");
    const size = parseInt(contentLength, 10);
    const fiftyMB = 50 * 1024 * 1024;
    assert.ok(size > fiftyMB, `Expected > 50 MB, got ${(size / 1024 / 1024).toFixed(1)} MB`);
  });

  it("first bytes match TFLite flatbuffer magic", async (t) => {
    if (!networkAvailable) return t.skip("no network");

    // Download just the first 8 bytes via Range header
    const rangeRes = await fetch(LITERT_GEMMA_MODEL_URL, {
      headers: { Range: "bytes=4-7" },
      signal: AbortSignal.timeout(15000),
      redirect: "follow",
    });

    // Some servers may not support Range; fall back to full response
    const buf = new Uint8Array(await rangeRes.arrayBuffer());
    const magic = rangeRes.status === 206 ? buf.slice(0, 4) : buf.slice(4, 8);

    assert.deepEqual(
      Array.from(magic),
      Array.from(TFLITE_MAGIC),
      `Expected TFLite magic bytes "TFL3", got [${Array.from(magic).map(b => '0x' + b.toString(16).padStart(2, '0')).join(', ')}]`
    );
  });
});
