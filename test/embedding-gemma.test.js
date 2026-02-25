/**
 * Integration tests for EmbeddingGemma via Transformers.js.
 *
 * These run in Node.js (no browser required) and verify that:
 *   1. The model loads successfully
 *   2. Single-text embedding returns the expected shape (768d)
 *   3. Batch embedding works and returns consistent results
 *   4. Semantically similar texts produce higher cosine similarity
 *   5. Embedding vectors are normalized
 *
 * Uses Node.js built-in test runner (node --test).
 * CI-friendly: downloads the q4 ONNX model (~75 MB) on first run.
 * Skips gracefully when network is unavailable.
 */

import { describe, it, before } from "node:test";
import assert from "node:assert/strict";
import { AutoTokenizer, AutoModel, env } from "@huggingface/transformers";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX";
const TASK_PREFIX = "task: classification | query: ";
const EXPECTED_DIM = 768;

env.allowLocalModels = false;

// ---------------------------------------------------------------------------
// Network check
// ---------------------------------------------------------------------------

async function checkNetwork() {
  try {
    const res = await fetch("https://huggingface.co/api/whoami-v2", {
      method: "HEAD",
      signal: AbortSignal.timeout(5000),
    });
    return true;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function l2Norm(vec) {
  let sum = 0;
  for (const v of vec) sum += v * v;
  return Math.sqrt(sum);
}

async function embed(tokenizer, model, texts) {
  const prefixed = texts.map(t => TASK_PREFIX + t);
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
// Tests
// ---------------------------------------------------------------------------

describe("EmbeddingGemma via Transformers.js", { timeout: 300_000 }, () => {
  let tokenizer;
  let model;
  let networkAvailable;

  before(async () => {
    networkAvailable = await checkNetwork();
    if (!networkAvailable) {
      console.log("Skipping EmbeddingGemma tests — no network access to download model.");
      return;
    }

    // Load model — CPU/WASM in Node.js (no WebGPU)
    [tokenizer, model] = await Promise.all([
      AutoTokenizer.from_pretrained(MODEL_ID),
      AutoModel.from_pretrained(MODEL_ID, {
        device: "cpu",
        dtype: "q4",
      }),
    ]);
  });

  it("produces a 768-dimensional embedding for a single text", async (t) => {
    if (!networkAvailable) return t.skip("no network");
    const [vec] = await embed(tokenizer, model, ["Hello world"]);
    assert.equal(vec.length, EXPECTED_DIM, `Expected ${EXPECTED_DIM}d, got ${vec.length}d`);
    assert.ok(vec.every(v => Number.isFinite(v)), "All values should be finite");
  });

  it("returns one embedding per text in batch mode", async (t) => {
    if (!networkAvailable) return t.skip("no network");
    const texts = ["cats", "dogs", "fish"];
    const vecs = await embed(tokenizer, model, texts);
    assert.equal(vecs.length, texts.length, `Expected ${texts.length} embeddings`);
    for (const vec of vecs) {
      assert.equal(vec.length, EXPECTED_DIM);
    }
  });

  it("gives higher similarity for semantically related texts", async (t) => {
    if (!networkAvailable) return t.skip("no network");
    const [vecCat, vecKitten, vecCar] = await embed(tokenizer, model, [
      "A cute cat sitting on a couch",
      "A kitten resting on the sofa",
      "The stock market crashed today",
    ]);

    const simRelated = cosineSimilarity(vecCat, vecKitten);
    const simUnrelated = cosineSimilarity(vecCat, vecCar);

    assert.ok(
      simRelated > simUnrelated,
      `Related sim (${simRelated.toFixed(4)}) should exceed unrelated sim (${simUnrelated.toFixed(4)})`
    );
  });

  it("produces approximately unit-norm vectors", async (t) => {
    if (!networkAvailable) return t.skip("no network");
    const [vec] = await embed(tokenizer, model, ["test normalization"]);
    const norm = l2Norm(vec);
    assert.ok(
      Math.abs(norm - 1.0) < 0.15,
      `Expected approximately unit norm, got ${norm.toFixed(4)}`
    );
  });

  it("returns deterministic results for the same input", async (t) => {
    if (!networkAvailable) return t.skip("no network");
    const text = "reproducibility test";
    const [vec1] = await embed(tokenizer, model, [text]);
    const [vec2] = await embed(tokenizer, model, [text]);
    const sim = cosineSimilarity(vec1, vec2);
    assert.ok(sim > 0.999, `Same input should produce near-identical vectors (sim=${sim.toFixed(6)})`);
  });
});
