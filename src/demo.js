/**
 * Vectory Embedding Playground — demo page.
 *
 * Uses the extension's internal messaging (chrome.runtime.sendMessage) to
 * call the same EmbeddingGemma backend that powers window.Embedder.
 */

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------

const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const textA = document.getElementById('textA');
const textB = document.getElementById('textB');
const compareBtn = document.getElementById('compareBtn');
const simResult = document.getElementById('simResult');
const simScore = document.getElementById('simScore');
const simBarFill = document.getElementById('simBarFill');
const scatterInput = document.getElementById('scatterInput');
const addBtn = document.getElementById('addBtn');
const tagList = document.getElementById('tagList');
const canvas = document.getElementById('scatterCanvas');
const ctx = canvas.getContext('2d');

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const entries = []; // { text, embedding }
let ready = false;

// ---------------------------------------------------------------------------
// Preset data
// ---------------------------------------------------------------------------

const PRESETS = {
  animals: [
    "A golden retriever playing fetch in the park",
    "A tabby cat sleeping on a sunny windowsill",
    "A dolphin leaping out of the ocean",
    "An eagle soaring above the mountains",
    "A rabbit hopping through a meadow",
    "A parrot mimicking human speech",
  ],
  tech: [
    "Machine learning model training on GPUs",
    "JavaScript framework for building web apps",
    "Quantum computing breakthrough announced",
    "Open source database migration tool",
    "Neural network architecture for NLP",
    "CSS grid layout for responsive design",
  ],
  emotions: [
    "Feeling overjoyed after getting great news",
    "Deep sadness and grief after a loss",
    "Burning with anger at an injustice",
    "Peaceful calm while watching the sunset",
    "Nervous anxiety before a big presentation",
    "Overwhelming love for family and friends",
  ],
  mixed: [
    "A chef preparing a gourmet Italian dinner",
    "Solving differential equations in calculus class",
    "Rock climbing on a steep mountain face",
    "A symphony orchestra performing Beethoven",
    "Debugging a memory leak in production",
    "A child building a sandcastle at the beach",
  ],
};

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

function simColor(sim) {
  if (sim > 0.7) return '#34d399';
  if (sim > 0.4) return '#fbbf24';
  return '#f87171';
}

/** Simple 2-component PCA via power iteration. */
function pca2D(vectors) {
  if (vectors.length < 2) return vectors.map(() => [0, 0]);

  const n = vectors.length;
  const d = vectors[0].length;

  // Center
  const mean = new Float64Array(d);
  for (const v of vectors) for (let i = 0; i < d; i++) mean[i] += v[i];
  for (let i = 0; i < d; i++) mean[i] /= n;

  const centered = vectors.map(v => v.map((x, i) => x - mean[i]));

  // Power iteration for top-2 eigenvectors of X^T X
  function topEigenvector(data, deflate) {
    let vec = new Float64Array(d);
    for (let i = 0; i < d; i++) vec[i] = Math.random() - 0.5;

    for (let iter = 0; iter < 50; iter++) {
      const proj = data.map(row => {
        let s = 0; for (let i = 0; i < d; i++) s += row[i] * vec[i]; return s;
      });
      const next = new Float64Array(d);
      for (let j = 0; j < n; j++) for (let i = 0; i < d; i++) next[i] += proj[j] * data[j][i];
      let norm = 0; for (let i = 0; i < d; i++) norm += next[i] * next[i];
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < d; i++) vec[i] = next[i] / norm;
    }
    return vec;
  }

  const pc1 = topEigenvector(centered);

  // Deflate
  const deflated = centered.map(row => {
    let proj = 0; for (let i = 0; i < d; i++) proj += row[i] * pc1[i];
    return row.map((x, i) => x - proj * pc1[i]);
  });

  const pc2 = topEigenvector(deflated);

  // Project
  return centered.map(row => {
    let x = 0, y = 0;
    for (let i = 0; i < d; i++) { x += row[i] * pc1[i]; y += row[i] * pc2[i]; }
    return [x, y];
  });
}

// ---------------------------------------------------------------------------
// Extension messaging
// ---------------------------------------------------------------------------

function sendMsg(type, payload) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ type, ...payload }, resolve);
  });
}

async function embedTexts(texts) {
  return sendMsg('VECTORY_EMBED', { task: 'embedBatch', texts });
}

async function embedSingle(text) {
  return sendMsg('VECTORY_EMBED', { task: 'embed', text });
}

// ---------------------------------------------------------------------------
// Init model
// ---------------------------------------------------------------------------

async function init() {
  setStatus('loading', 'Checking model availability...');

  const availability = await sendMsg('VECTORY_AVAILABILITY', { options: { modality: 'text' } });

  if (availability === 'available') {
    setStatus('ready', 'EmbeddingGemma ready');
    enableUI();
    return;
  }

  if (availability === 'unavailable') {
    setStatus('error', 'Embedder unavailable on this device');
    return;
  }

  setStatus('loading', 'Downloading EmbeddingGemma (~75 MB)...');

  const result = await sendMsg('VECTORY_CREATE', { options: { modality: 'text' } });

  if (result?.success) {
    setStatus('ready', 'EmbeddingGemma ready');
    enableUI();
  } else {
    setStatus('error', `Load failed: ${result?.error || 'unknown'}`);
  }
}

function setStatus(state, text) {
  statusDot.className = 'status-dot ' + state;
  statusText.textContent = text;
}

function enableUI() {
  ready = true;
  compareBtn.disabled = false;
  addBtn.disabled = false;
}

// ---------------------------------------------------------------------------
// Similarity
// ---------------------------------------------------------------------------

compareBtn.addEventListener('click', async () => {
  const a = textA.value.trim();
  const b = textB.value.trim();
  if (!a || !b) return;

  compareBtn.disabled = true;
  compareBtn.textContent = 'Computing...';

  const vecs = await embedTexts([a, b]);

  if (vecs?.error) {
    simScore.textContent = 'Error';
    simResult.classList.add('visible');
    compareBtn.disabled = false;
    compareBtn.textContent = 'Compare';
    return;
  }

  const sim = cosineSimilarity(vecs[0], vecs[1]);
  const pct = Math.max(0, Math.min(100, sim * 100));

  simScore.textContent = sim.toFixed(4);
  simScore.style.color = simColor(sim);
  simBarFill.style.width = pct + '%';
  simBarFill.style.background = simColor(sim);
  simResult.classList.add('visible');

  compareBtn.disabled = false;
  compareBtn.textContent = 'Compare';
});

// ---------------------------------------------------------------------------
// Scatter plot
// ---------------------------------------------------------------------------

async function addText(text) {
  if (!text.trim()) return;

  const vec = await embedSingle(text.trim());
  if (vec?.error) return;

  entries.push({ text: text.trim(), embedding: vec });
  renderTags();
  renderScatter();
}

function removeEntry(idx) {
  entries.splice(idx, 1);
  renderTags();
  renderScatter();
}

function renderTags() {
  tagList.innerHTML = '';
  entries.forEach((e, i) => {
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.innerHTML = `${escapeHtml(truncate(e.text, 30))}<span class="remove" data-idx="${i}">&times;</span>`;
    tagList.appendChild(tag);
  });

  tagList.querySelectorAll('.remove').forEach(el => {
    el.addEventListener('click', () => removeEntry(parseInt(el.dataset.idx)));
  });
}

function truncate(s, n) {
  return s.length > n ? s.slice(0, n) + '...' : s;
}

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderScatter() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const w = rect.width;
  const h = rect.height;

  ctx.clearRect(0, 0, w, h);

  if (entries.length === 0) {
    ctx.fillStyle = '#71717a';
    ctx.font = '13px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Add texts to see them plotted here', w / 2, h / 2);
    return;
  }

  if (entries.length === 1) {
    drawPoint(w / 2, h / 2, entries[0].text, 0);
    return;
  }

  const points = pca2D(entries.map(e => e.embedding));

  // Compute bounds
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of points) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const pad = 40;
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  points.forEach(([px, py], i) => {
    const x = pad + ((px - minX) / rangeX) * (w - 2 * pad);
    const y = pad + ((py - minY) / rangeY) * (h - 2 * pad);
    drawPoint(x, y, entries[i].text, i);
  });
}

const COLORS = [
  '#818cf8', '#34d399', '#fbbf24', '#f87171',
  '#a78bfa', '#2dd4bf', '#fb923c', '#f472b6',
  '#60a5fa', '#4ade80', '#facc15', '#e879f9',
];

function drawPoint(x, y, text, idx) {
  const color = COLORS[idx % COLORS.length];

  // Dot
  ctx.beginPath();
  ctx.arc(x, y, 5, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();

  // Glow
  ctx.beginPath();
  ctx.arc(x, y, 5, 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.globalAlpha = 0.3;
  ctx.stroke();
  ctx.globalAlpha = 1;

  // Label
  ctx.font = '11px system-ui, sans-serif';
  ctx.fillStyle = '#e4e4e7';
  ctx.textAlign = 'center';
  ctx.fillText(truncate(text, 25), x, y - 10);
}

addBtn.addEventListener('click', () => {
  const text = scatterInput.value.trim();
  if (text) {
    addText(text);
    scatterInput.value = '';
  }
});

scatterInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && ready) {
    const text = scatterInput.value.trim();
    if (text) {
      addText(text);
      scatterInput.value = '';
    }
  }
});

// Presets
document.querySelectorAll('.preset-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    if (!ready) return;
    const key = btn.dataset.preset;
    const texts = PRESETS[key];
    if (!texts) return;

    btn.disabled = true;
    btn.textContent = 'Loading...';

    // Clear existing
    entries.length = 0;
    renderTags();

    // Batch embed all preset texts
    const vecs = await embedTexts(texts);
    if (!vecs?.error) {
      for (let i = 0; i < texts.length; i++) {
        entries.push({ text: texts[i], embedding: vecs[i] });
      }
      renderTags();
      renderScatter();
    }

    btn.disabled = false;
    btn.textContent = key.charAt(0).toUpperCase() + key.slice(1);
  });
});

// Resize handler
window.addEventListener('resize', () => renderScatter());

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

init();
