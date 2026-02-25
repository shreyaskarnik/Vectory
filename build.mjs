#!/usr/bin/env node
/**
 * Build script for Vectory.
 *
 * 1. Vite bundles background.js (inlines Transformers.js + ONNX runtime)
 * 2. Copies static extension files to dist/
 * 3. Copies MediaPipe lib/ directory if present (for USE backend)
 */
import { build } from "vite";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const src = (...parts) => resolve(__dirname, "src", ...parts);
const dist = (...parts) => resolve(__dirname, "dist", ...parts);

// 1. Bundle background.js with Transformers.js
await build({ configFile: resolve(__dirname, "vite.config.js") });

// 2. Copy static files
const staticFiles = [
  "manifest.json",
  "content.js",
  "embedder-api.js",
  "offscreen.html",
  "offscreen.js",
  "demo.html",
  "demo.js",
];

for (const file of staticFiles) {
  fs.copyFileSync(src(file), dist(file));
}

// 3. Copy MediaPipe lib/ if it exists (for USE fallback)
const srcLib = src("lib");
if (fs.existsSync(srcLib)) {
  fs.cpSync(srcLib, dist("lib"), { recursive: true });
}

console.log("\nVectory build complete → dist/");
