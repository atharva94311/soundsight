// Runs the browser classifier over feature tensors from test_inference_parity.py
// and prints logits + probabilities. Invoked by the test.
//
//   node infer_harness.mjs <feats.json> <out.json>
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const cnnPath = resolve(here, '../../vas3d/js/audio/cnn.js');

// cnn.js decodes its weights with atob(), which node exposes globally from v16.
const { forward, probabilities, CLASSES, INPUT_SHAPE } = await import(pathToFileURL(cnnPath).href);

const [, , inPath, outPath] = process.argv;
const feats = JSON.parse(readFileSync(inPath, 'utf8'));

const results = feats.map((f) => {
  const logits = forward(Float32Array.from(f));
  return {
    logits: Array.from(logits),
    probs: Array.from(probabilities(logits)),
  };
});

writeFileSync(outPath, JSON.stringify({ classes: CLASSES, input_shape: INPUT_SHAPE, results }));
