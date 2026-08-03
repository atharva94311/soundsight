// Runs the browser front-end over signals produced by test_parity.py and prints
// the resulting features as JSON. Invoked by the test; not useful on its own.
//
//   node parity_harness.mjs <signals.json> <out.json>
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const melPath = resolve(here, '../../vas3d/js/audio/mel.js');
const { features, logMel, mfcc, SHAPE } = await import(pathToFileURL(melPath).href);

const [, , inPath, outPath] = process.argv;
const signals = JSON.parse(readFileSync(inPath, 'utf8'));

const out = {};
for (const [name, samples] of Object.entries(signals)) {
  const x = Float64Array.from(samples);
  out[name] = {
    logmel: Array.from(logMel(x)),
    mfcc: Array.from(mfcc(x)),
    features: Array.from(features(x)),
  };
}
writeFileSync(outPath, JSON.stringify({ shape: SHAPE, results: out }));
