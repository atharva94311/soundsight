// ============================================================================
//  infer.js — one window of audio in, class probabilities out.
//
//  Kept separate from listener.js so the exact same code runs in the inference
//  worker and in the main-thread fallback. Pure: no DOM, no state.
// ============================================================================
import { DSP } from './dsp-config.js';
import { features, normalize } from './mel.js';
import { classify, NORM } from './cnn.js';

/**
 * @param {Float32Array} samples  DSP.n_samples mono floats at DSP.sample_rate
 * @returns {{probs: Float32Array, spectrum: number[]}}
 */
export function infer(samples) {
  const feat = normalize(features(samples), NORM.mean, NORM.std);
  return { probs: classify(feat), spectrum: spectrumFor(feat) };
}

/**
 * Collapse the (bins x frames) feature into the 24-bin 0..1 vector the twin's
 * spectrum display expects, so the on-screen spectrum is the audio that was
 * actually heard rather than the synthesised formants from config.js.
 */
export function spectrumFor(feat, outBins = 24) {
  const bins = DSP.n_bins, frames = DSP.n_frames;

  // Average the last few frames — a single frame is visually noisy.
  const take = Math.min(6, frames);
  const band = new Float32Array(bins);
  for (let m = 0; m < bins; m++) {
    let acc = 0;
    for (let t = frames - take; t < frames; t++) acc += feat[m * frames + t];
    band[m] = acc / take;
  }

  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < bins; i++) { if (band[i] < lo) lo = band[i]; if (band[i] > hi) hi = band[i]; }
  const span = hi - lo || 1;

  const out = new Array(outBins);
  for (let i = 0; i < outBins; i++) {
    // bins (40) -> outBins (24) by linear interpolation along the mel axis
    const pos = (i / (outBins - 1)) * (bins - 1);
    const j = Math.floor(pos);
    const frac = pos - j;
    const a = band[j], b = band[Math.min(j + 1, bins - 1)];
    out[i] = Math.max(0, Math.min(1, ((a + (b - a) * frac) - lo) / span));
  }
  return out;
}
