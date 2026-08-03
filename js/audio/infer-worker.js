// ============================================================================
//  infer-worker.js — runs the front-end and the network off the main thread.
//
//  One inference costs ~22 ms. Four times a second on the main thread that is a
//  dropped frame or two per hop, which is very visible in a 3D scene that is
//  otherwise smooth. Here it costs the render loop nothing.
//
//  Module worker: `new Worker(url, { type: 'module' })`.
// ============================================================================
import { infer } from './infer.js';

self.onmessage = (e) => {
  const { samples, seq } = e.data;
  const { probs, spectrum } = infer(samples);
  // probs is a fresh Float32Array each call, so it is safe to transfer.
  self.postMessage({ probs, spectrum, seq }, [probs.buffer]);
};

// Tell the main thread we have finished loading the model, so it can report a
// real failure (missing weights, bad module) instead of silently never firing.
self.postMessage({ ready: true });
