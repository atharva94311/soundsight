// ============================================================================
//  mel.js — the browser half of the audio front-end.
//
//  This is a line-for-line mirror of ml/vas_ml/features.py. The model was trained
//  on features produced by that file; if this one computes them even slightly
//  differently the network still runs and still returns confident-looking numbers,
//  it is just wrong about what it heard. ml/tests/test_parity.py runs the same
//  audio through both and asserts they agree, so run it after touching this.
//
//  Everything is float64 (JS numbers) to match numpy, and cast to float32 only at
//  the very end, exactly as features.py does.
// ============================================================================
import { DSP } from './dsp-config.js';

// ---------------------------------------------------------------------------
//  FFT — iterative radix-2 Cooley-Tukey, in-place, with precomputed tables.
//  n_fft is a power of two by construction (see DSP.n_fft).
// ---------------------------------------------------------------------------
function buildFFT(n) {
  if ((n & (n - 1)) !== 0) throw new Error(`n_fft must be a power of two, got ${n}`);

  // bit-reversal permutation table
  const rev = new Uint32Array(n);
  const bits = Math.log2(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
    rev[i] = r;
  }

  // twiddles per stage, flattened: for len=2,4,..,n we store len/2 (cos,sin) pairs
  const cos = [], sin = [];
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const c = new Float64Array(half), s = new Float64Array(half);
    for (let j = 0; j < half; j++) {
      const ang = (-2 * Math.PI * j) / len;
      c[j] = Math.cos(ang);
      s[j] = Math.sin(ang);
    }
    cos.push(c); sin.push(s);
  }

  /** In-place complex FFT. re/im are Float64Array(n); im is zeroed for real input. */
  return function fft(re, im) {
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    let stage = 0;
    for (let len = 2; len <= n; len <<= 1, stage++) {
      const half = len >> 1;
      const c = cos[stage], s = sin[stage];
      for (let i = 0; i < n; i += len) {
        for (let j = 0; j < half; j++) {
          const a = i + j, b = a + half;
          const tr = re[b] * c[j] - im[b] * s[j];
          const ti = re[b] * s[j] + im[b] * c[j];
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr;        im[a] += ti;
        }
      }
    }
  };
}

// ---------------------------------------------------------------------------
//  Mel scale — HTK convention, same as features.py and vas3d/js/config.js
// ---------------------------------------------------------------------------
export const hzToMel = (f) => 2595 * Math.log10(1 + f / 700);
export const melToHz = (m) => 700 * (10 ** (m / 2595) - 1);

/**
 * Triangular mel filters, unit peak, stored sparsely as {start, weights} per band.
 *
 * The dense form is ~97% zeros: 40 bands x 513 bins, of which each band touches
 * only a contiguous handful. Iterating the dense matrix costs 20k multiply-adds
 * per frame against 49 frames, and almost all of it is multiplying by zero. Each
 * triangle is contiguous in k, so a start index plus a short weight run is exact,
 * not an approximation. (firmware/esp32/vas_dsp.c stores it the same way.)
 */
function melFilterbank() {
  const nOut = DSP.n_fft / 2 + 1;
  const edges = new Float64Array(DSP.n_mels + 2);
  const m0 = hzToMel(DSP.fmin), m1 = hzToMel(DSP.fmax);
  for (let i = 0; i < DSP.n_mels + 2; i++) {
    const mel = m0 + ((m1 - m0) * i) / (DSP.n_mels + 1);      // linspace inclusive
    edges[i] = (melToHz(mel) * DSP.n_fft) / DSP.sample_rate;  // fractional FFT bins
  }

  const bands = [];
  for (let m = 0; m < DSP.n_mels; m++) {
    const lo = edges[m], mid = edges[m + 1], hi = edges[m + 2];
    const vals = [];
    let start = -1;
    for (let k = 0; k < nOut; k++) {
      let v = 0;
      if (mid > lo && k > lo && k <= mid) v = (k - lo) / (mid - lo);
      else if (hi > mid && k > mid && k < hi) v = (hi - k) / (hi - mid);
      if (v > 0) {
        if (start < 0) start = k;
        vals.push(v);
      } else if (start >= 0) {
        break;                        // triangles are contiguous; past the far edge
      }
    }
    bands.push({ start: start < 0 ? 0 : start, w: Float64Array.from(vals) });
  }
  return bands;
}

/** Periodic Hann, matching scipy sym=False. */
function hann(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

/** Orthonormal DCT-II, (nOut x nIn) flat. */
function dct2Matrix(nIn, nOut) {
  const m = new Float64Array(nOut * nIn);
  for (let k = 0; k < nOut; k++) {
    const scale = Math.sqrt(2 / nIn) * (k === 0 ? Math.sqrt(0.5) : 1);
    for (let n = 0; n < nIn; n++) {
      m[k * nIn + n] = Math.cos((Math.PI * k * (2 * n + 1)) / (2 * nIn)) * scale;
    }
  }
  return m;
}

const N_OUT = DSP.n_fft / 2 + 1;
const FFT = buildFFT(DSP.n_fft);
const FB = melFilterbank();
const WIN = hann(DSP.win_length);
const DCT = dct2Matrix(DSP.n_mels, DSP.n_mfcc);

// Scratch buffers — reused across calls so a 4 Hz inference loop allocates nothing.
const _re = new Float64Array(DSP.n_fft);
const _im = new Float64Array(DSP.n_fft);
const _power = new Float64Array(N_OUT);
const _logmel = new Float64Array(DSP.n_mels * DSP.n_frames);

/**
 * (n_mels x n_frames) log-mel spectrogram, row-major, as a flat Float64Array.
 * `x` is mono float samples at DSP.sample_rate; short input is zero-padded,
 * long input is truncated — same as frame_signal() in features.py.
 */
export function logMel(x) {
  _logmel.fill(0);
  for (let t = 0; t < DSP.n_frames; t++) {
    const off = t * DSP.hop_length;

    _re.fill(0); _im.fill(0);
    for (let i = 0; i < DSP.win_length; i++) {
      const s = off + i < x.length ? x[off + i] : 0;
      _re[i] = s * WIN[i];
    }
    FFT(_re, _im);

    for (let k = 0; k < N_OUT; k++) _power[k] = _re[k] * _re[k] + _im[k] * _im[k];

    for (let m = 0; m < DSP.n_mels; m++) {
      const band = FB[m], w = band.w, s = band.start;
      let acc = 0;
      for (let k = 0; k < w.length; k++) acc += _power[s + k] * w[k];
      _logmel[m * DSP.n_frames + t] = Math.log(acc + DSP.log_floor);
    }
  }
  return _logmel;
}

/** (n_mfcc x n_frames) flat Float64Array. */
export function mfcc(x) {
  const lm = logMel(x);
  const out = new Float64Array(DSP.n_mfcc * DSP.n_frames);
  for (let k = 0; k < DSP.n_mfcc; k++) {
    for (let t = 0; t < DSP.n_frames; t++) {
      let acc = 0;
      for (let m = 0; m < DSP.n_mels; m++) acc += DCT[k * DSP.n_mels + m] * lm[m * DSP.n_frames + t];
      out[k * DSP.n_frames + t] = acc;
    }
  }
  return out;
}

/** (n_bins x n_frames) Float32Array, unnormalised — mirrors features(). */
export function features(x) {
  const src = DSP.feature === 'mfcc' ? mfcc(x) : logMel(x);
  return Float32Array.from(src);
}

/**
 * Global (scalar) standardisation, in place.
 * Not per-example, for the same reason as features.py: normalising each window by
 * its own statistics turns a quiet room into a confident detection.
 */
export function normalize(feat, mean, std) {
  for (let i = 0; i < feat.length; i++) feat[i] = (feat[i] - mean) / std;
  return feat;
}

export const SHAPE = { bins: DSP.n_bins, frames: DSP.n_frames };
