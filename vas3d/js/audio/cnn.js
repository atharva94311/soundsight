// ============================================================================
//  cnn.js — the classifier's forward pass, by hand.
//
//  The exported graph is only conv / ReLU / global-average-pool / linear (batchnorm
//  was folded into the convolutions at export), which is few enough ops to run
//  directly. That keeps the twin dependency-free: no ONNX runtime, no wasm blob,
//  no build step — the same reason three.js is vendored rather than imported.
//
//  Checked against PyTorch by ml/tests/test_inference_parity.py.
// ============================================================================
import { GRAPH, WEIGHTS } from './model-weights.js';

/** A layer's activation: channel-major (c, h, w) in one Float32Array. */
function tensor(c, h, w) {
  return { c, h, w, d: new Float32Array(c * h * w) };
}

// Scratch tensors are allocated once on first run and reused, so the 4 Hz
// inference loop does not churn the GC.
let _scratch = null;

function planShapes() {
  const [bins, frames] = GRAPH.input_shape;
  const shapes = [{ c: 1, h: bins, w: frames }];
  let cur = shapes[0];
  for (const op of GRAPH.ops) {
    if (op.type === 'conv') {
      cur = {
        c: op.cout,
        h: Math.floor((cur.h + 2 * op.ph - op.kh) / op.sh) + 1,
        w: Math.floor((cur.w + 2 * op.pw - op.kw) / op.sw) + 1,
      };
    } else if (op.type === 'gap') {
      cur = { c: cur.c, h: 1, w: 1 };
    } else if (op.type === 'linear') {
      cur = { c: op.cout, h: 1, w: 1 };
    }
    shapes.push(cur);
  }
  return shapes;
}

function ensureScratch() {
  if (_scratch) return _scratch;
  const shapes = planShapes();
  _scratch = shapes.map((s) => tensor(s.c, s.h, s.w));
  return _scratch;
}

/**
 * Pointwise (1x1) convolution — about 85% of the network's multiply-accumulates.
 *
 * With no kernel extent and no padding it is a channel mix at each spatial
 * position, so it runs as bias-fill then accumulate along contiguous planes.
 * That inner loop is a straight strided read of two typed arrays, which the JIT
 * handles roughly an order of magnitude better than the general path's
 * per-element bounds checks.
 */
function conv1x1(src, dst, op) {
  const P = dst.h * dst.w;
  const cin = src.c, cout = op.cout;
  const W = WEIGHTS, wOff = op.w_off, bOff = op.b_off;
  const sd = src.d, dd = dst.d;

  for (let oc = 0; oc < cout; oc++) {
    const dBase = oc * P;
    dd.fill(W[bOff + oc], dBase, dBase + P);
    const wBase = wOff + oc * cin;

    for (let ic = 0; ic < cin; ic++) {
      const w = W[wBase + ic];
      if (w === 0) continue;
      const sBase = ic * P;
      for (let p = 0; p < P; p++) dd[dBase + p] += sd[sBase + p] * w;
    }
    if (op.relu) {
      for (let p = dBase; p < dBase + P; p++) if (dd[p] < 0) dd[p] = 0;
    }
  }
}

/**
 * General grouped convolution. `groups === cin === cout` is the depthwise case.
 *
 * The padding bounds are hoisted out of the innermost loop: for a given output
 * row and column the valid kernel range is fixed, so it is computed once instead
 * of being re-tested per tap.
 */
function convKxK(src, dst, op) {
  const { kh, kw, sh, sw, ph, pw, groups, cout } = op;
  const cinG = (src.c / groups) | 0;
  const coutG = (cout / groups) | 0;
  const W = WEIGHTS, wOff = op.w_off, bOff = op.b_off;
  const sH = src.h, sW = src.w, dH = dst.h, dW = dst.w;
  const sPlane = sH * sW;
  const kSize = cinG * kh * kw;
  const sd = src.d, dd = dst.d;

  for (let oc = 0; oc < cout; oc++) {
    const grp = (oc / coutG) | 0;
    const bias = W[bOff + oc];
    const wBase = wOff + oc * kSize;
    const dBase = oc * dH * dW;

    for (let oy = 0; oy < dH; oy++) {
      const iy0 = oy * sh - ph;
      const aLo = iy0 < 0 ? -iy0 : 0;
      const aHi = iy0 + kh > sH ? sH - iy0 : kh;

      for (let ox = 0; ox < dW; ox++) {
        const ix0 = ox * sw - pw;
        const bLo = ix0 < 0 ? -ix0 : 0;
        const bHi = ix0 + kw > sW ? sW - ix0 : kw;
        let acc = bias;

        for (let ic = 0; ic < cinG; ic++) {
          const sBase = (grp * cinG + ic) * sPlane;
          const kBase = wBase + ic * kh * kw;
          for (let a = aLo; a < aHi; a++) {
            const rBase = sBase + (iy0 + a) * sW + ix0;
            const kRow = kBase + a * kw;
            for (let b = bLo; b < bHi; b++) acc += sd[rBase + b] * W[kRow + b];
          }
        }
        dd[dBase + oy * dW + ox] = op.relu ? (acc > 0 ? acc : 0) : acc;
      }
    }
  }
}

function conv2d(src, dst, op) {
  // The exported graph only ever emits 1x1 as stride-1, unpadded, ungrouped.
  // Anything else falls through to the general path rather than being assumed.
  if (op.kh === 1 && op.kw === 1 && op.sh === 1 && op.sw === 1 &&
      op.ph === 0 && op.pw === 0 && op.groups === 1) {
    conv1x1(src, dst, op);
  } else {
    convKxK(src, dst, op);
  }
}

function globalAvgPool(src, dst) {
  const plane = src.h * src.w;
  for (let c = 0; c < src.c; c++) {
    let acc = 0;
    const base = c * plane;
    for (let i = 0; i < plane; i++) acc += src.d[base + i];
    dst.d[c] = acc / plane;
  }
}

function linear(src, dst, op) {
  const W = WEIGHTS, wOff = op.w_off, bOff = op.b_off;
  for (let o = 0; o < op.cout; o++) {
    let acc = W[bOff + o];
    const base = wOff + o * op.cin;
    for (let i = 0; i < op.cin; i++) acc += W[base + i] * src.d[i];
    dst.d[o] = op.relu ? (acc > 0 ? acc : 0) : acc;
  }
}

/**
 * Normalised (bins x frames) Float32Array -> logits Float32Array(n_classes).
 * The input must already have been standardised with GRAPH.norm.
 */
export function forward(feat) {
  const bufs = ensureScratch();
  bufs[0].d.set(feat);

  for (let i = 0; i < GRAPH.ops.length; i++) {
    const op = GRAPH.ops[i];
    const src = bufs[i], dst = bufs[i + 1];
    if (op.type === 'conv') conv2d(src, dst, op);
    else if (op.type === 'gap') globalAvgPool(src, dst);
    else if (op.type === 'linear') linear(src, dst, op);
    else throw new Error(`unknown op ${op.type}`);
  }
  return bufs[bufs.length - 1].d;
}

/**
 * Softmax with the calibration temperature applied.
 *
 * The temperature was fitted on the validation set at training time. Without it
 * the network is overconfident and the 75% gate stops meaning "75% sure".
 */
export function probabilities(logits, temperature = GRAPH.temperature) {
  const n = logits.length;
  const out = new Float32Array(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    out[i] = logits[i] / temperature;
    if (out[i] > max) max = out[i];
  }
  let sum = 0;
  for (let i = 0; i < n; i++) { out[i] = Math.exp(out[i] - max); sum += out[i]; }
  for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

/** Convenience: normalised features -> probabilities. */
export function classify(feat) {
  return probabilities(forward(feat));
}

export const CLASSES = GRAPH.classes;
export const NORM = GRAPH.norm;
export const INPUT_SHAPE = GRAPH.input_shape;
