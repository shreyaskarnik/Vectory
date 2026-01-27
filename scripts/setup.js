const fs = require('fs');
const path = require('path');

const projectRoot = path.resolve(__dirname, '..');
const srcLib = path.join(projectRoot, 'src', 'lib');
const wasmDir = path.join(srcLib, 'wasm');

// Ensure directories exist
if (!fs.existsSync(srcLib)) fs.mkdirSync(srcLib, { recursive: true });
if (!fs.existsSync(wasmDir)) fs.mkdirSync(wasmDir, { recursive: true });

const nodeModules = path.join(projectRoot, 'node_modules', '@mediapipe', 'tasks-text');

// Files to copy
const filesToCopy = [
    { src: path.join(nodeModules, 'text_bundle.mjs'), dest: path.join(srcLib, 'tasks-text.js') },
    // Copy WASM files
    { src: path.join(nodeModules, 'wasm', 'text_wasm_internal.js'), dest: path.join(wasmDir, 'text_wasm_internal.js') },
    { src: path.join(nodeModules, 'wasm', 'text_wasm_internal.wasm'), dest: path.join(wasmDir, 'text_wasm_internal.wasm') },
    { src: path.join(nodeModules, 'wasm', 'text_wasm_nosimd_internal.js'), dest: path.join(wasmDir, 'text_wasm_nosimd_internal.js') },
    { src: path.join(nodeModules, 'wasm', 'text_wasm_nosimd_internal.wasm'), dest: path.join(wasmDir, 'text_wasm_nosimd_internal.wasm') },
];

console.log('Copying MediaPipe assets...');

filesToCopy.forEach(item => {
    if (fs.existsSync(item.src)) {
        fs.copyFileSync(item.src, item.dest);
        console.log(`Copied ${path.basename(item.src)} to ${path.relative(projectRoot, item.dest)}`);
    } else {
        console.error(`Source file not found: ${item.src}`);
        console.log('Please run "npm install" first.');
        process.exit(1);
    }
});

console.log('Setup complete.');
