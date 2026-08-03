"""Turns the trained checkpoint into something the browser and the ESP32 can run.

    ml/.venv/bin/python export.py

Produces:
  vas3d/js/audio/model-weights.js   float32 weights, base64, as an ES module
  firmware/esp32/model_weights.h    int16 weights + float scales, as a C header
  ml/artifacts/graph.json           the op list both runtimes walk

Batchnorm is folded into the preceding convolution here, so what ships is just
conv / ReLU / global-average-pool / linear. That is few enough ops to reimplement
by hand in JS and C, which is why there is no interpreter or runtime dependency on
either target.

The browser gets float32 because there is no reason not to. The firmware gets
int16 *weights* with float32 activations: half the flash of float weights, no
calibration step, and no measurable effect on the decision. int8 was tried first
and rejected — it flipped the gate decision on ~4.4% of real test windows, since
the error accumulates across ten layers. Full int8 activations would save RAM
too, but need a calibration pass and careful requantisation.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from vas_ml.config import (
    ARTIFACTS, CLASSES, CONF_THRESHOLD, DSP_CFG, INFER_HOP_S, REFRACTORY_S,
    REPO, SMOOTH_HITS, SMOOTH_WINDOW,
)
from vas_ml.model import build

JS_OUT = REPO / "vas3d" / "js" / "audio" / "model-weights.js"
C_OUT = REPO / "firmware" / "esp32" / "model_weights.h"


# ---------------------------------------------------------------------------
#  Batchnorm folding
# ---------------------------------------------------------------------------
def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray]:
    """W' = W * gamma/sqrt(var+eps);  b' = beta - gamma*mu/sqrt(var+eps)"""
    w = conv.weight.detach().double().numpy()
    gamma = bn.weight.detach().double().numpy()
    beta = bn.bias.detach().double().numpy()
    mu = bn.running_mean.detach().double().numpy()
    var = bn.running_var.detach().double().numpy()
    scale = gamma / np.sqrt(var + bn.eps)

    w_folded = w * scale[:, None, None, None]
    b_folded = beta - scale * mu
    if conv.bias is not None:
        b_folded = b_folded + conv.bias.detach().double().numpy() * scale
    return w_folded, b_folded


def flatten_graph(model: nn.Module) -> list[dict]:
    """Walk the model and emit a flat op list with folded weights attached."""
    ops: list[dict] = []
    mods = list(model.features.modules())
    i = 0
    while i < len(mods):
        m = mods[i]
        if isinstance(m, nn.Conv2d):
            bn = mods[i + 1]
            assert isinstance(bn, nn.BatchNorm2d), "expected conv immediately followed by bn"
            w, b = fold_conv_bn(m, bn)
            ops.append({
                "type": "conv",
                "cin": m.in_channels, "cout": m.out_channels,
                "kh": m.kernel_size[0], "kw": m.kernel_size[1],
                "sh": m.stride[0], "sw": m.stride[1],
                "ph": m.padding[0], "pw": m.padding[1],
                "groups": m.groups,
                "relu": True,
                "_w": w, "_b": b,
            })
            i += 2
            continue
        i += 1

    ops.append({"type": "gap"})
    fc = model.fc
    ops.append({
        "type": "linear",
        "cin": fc.in_features, "cout": fc.out_features,
        "relu": False,
        "_w": fc.weight.detach().double().numpy(),
        "_b": fc.bias.detach().double().numpy(),
    })
    return ops


# ---------------------------------------------------------------------------
#  Reference forward pass in numpy — the thing JS and C are checked against
# ---------------------------------------------------------------------------
def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, op: dict) -> np.ndarray:
    """x: (cin, h, w) -> (cout, oh, ow). Plain and slow; correctness is the point."""
    cin, H, W = x.shape
    kh, kw, sh, sw, ph, pw, g = op["kh"], op["kw"], op["sh"], op["sw"], op["ph"], op["pw"], op["groups"]
    cout = op["cout"]
    oh = (H + 2 * ph - kh) // sh + 1
    ow = (W + 2 * pw - kw) // sw + 1

    xp = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    out = np.zeros((cout, oh, ow), dtype=np.float64)
    cin_g, cout_g = cin // g, cout // g

    for oc in range(cout):
        grp = oc // cout_g
        acc = np.full((oh, ow), b[oc], dtype=np.float64)
        for ic in range(cin_g):
            src = xp[grp * cin_g + ic]
            k = w[oc, ic]
            for a in range(kh):
                for c in range(kw):
                    acc += k[a, c] * src[a: a + oh * sh: sh, c: c + ow * sw: sw]
        out[oc] = acc
    return out


def run_graph(ops: list[dict], feat: np.ndarray) -> np.ndarray:
    """feat: (bins, frames) normalised -> (n_classes,) logits."""
    x = feat[None].astype(np.float64)
    for op in ops:
        if op["type"] == "conv":
            x = conv2d(x, op["_w"], op["_b"], op)
            if op["relu"]:
                x = np.maximum(x, 0.0)
        elif op["type"] == "gap":
            x = x.mean(axis=(1, 2))
        elif op["type"] == "linear":
            x = op["_w"] @ x + op["_b"]
    return x


# ---------------------------------------------------------------------------
#  Emitters
# ---------------------------------------------------------------------------
def emit_js(ops: list[dict], ckpt: dict) -> Path:
    blob: list[np.ndarray] = []
    meta: list[dict] = []
    off = 0
    for op in ops:
        d = {k: v for k, v in op.items() if not k.startswith("_")}
        if "_w" in op:
            w = op["_w"].astype(np.float32).ravel()
            b = op["_b"].astype(np.float32).ravel()
            d["w_off"], d["w_len"] = off, len(w); off += len(w)
            d["b_off"], d["b_len"] = off, len(b); off += len(b)
            blob += [w, b]
        meta.append(d)

    flat = np.concatenate(blob).astype("<f4") if blob else np.zeros(0, "<f4")
    b64 = base64.b64encode(flat.tobytes()).decode("ascii")

    graph = {
        "ops": meta,
        "classes": ckpt["classes"],
        "norm": ckpt["norm"],
        "temperature": ckpt["temperature"],
        "input_shape": list(DSP_CFG.shape),
        "n_weights": int(flat.size),
    }

    lines = [
        "// GENERATED by ml/export.py — do not edit by hand.",
        "//",
        "// Weights for the sound classifier, float32, base64-encoded so the twin stays",
        "// a set of static files with no fetch and no build step (same reasoning as the",
        "// vendored three.js). Decoded once at module load.",
        "",
        f"export const GRAPH = {json.dumps(graph, indent=2)};",
        "",
        f'const B64 = "{b64}";',
        "",
        "function decode(b64) {",
        "  const bin = atob(b64);",
        "  const bytes = new Uint8Array(bin.length);",
        "  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);",
        "  return new Float32Array(bytes.buffer);",
        "}",
        "",
        "export const WEIGHTS = decode(B64);",
        "",
    ]
    JS_OUT.parent.mkdir(parents=True, exist_ok=True)
    JS_OUT.write_text("\n".join(lines))
    return JS_OUT


def emit_c(ops: list[dict], ckpt: dict) -> Path:
    """int16 weights, per-output-channel scales, float32 biases.

    int8 was the first choice and it was the wrong trade. Measured on real test
    windows, int8 per-channel weights changed the model's gate decision on ~4.4% of
    windows — the raw probability error is small on average but the network is ten
    layers deep and the error accumulates. int16 drops that to noise for 35 kB more
    flash, on a part with 8 MB. The flash was never the binding constraint; RAM and
    decision fidelity are.
    """
    chunks: list[str] = []
    meta: list[str] = []
    total_bytes = 0

    QMAX = 32767   # int16

    def carr(name: str, arr: np.ndarray, ctype: str) -> str:
        vals = ", ".join(
            (f"{v:d}" if ctype.startswith("int") else f"{v:.8g}f") for v in arr.ravel().tolist()
        )
        return f"static const {ctype} {name}[{arr.size}] = {{{vals}}};\n"

    for i, op in enumerate(ops):
        if "_w" not in op:
            continue
        w = op["_w"]
        # Per-output-channel symmetric quantisation: each filter gets its own scale,
        # which matters because depthwise filters vary hugely in magnitude — layer 7
        # of this model spans a 27x range across its channels.
        flat = w.reshape(w.shape[0], -1)
        scale = np.abs(flat).max(axis=1) / float(QMAX)
        scale[scale == 0] = 1e-12
        q = np.clip(np.round(flat / scale[:, None]), -QMAX, QMAX).astype(np.int16)

        chunks.append(carr(f"w{i}_q", q, "int16_t"))
        chunks.append(carr(f"w{i}_s", scale.astype(np.float32), "float"))
        chunks.append(carr(f"b{i}", op["_b"].astype(np.float32), "float"))
        total_bytes += q.size * 2 + scale.size * 4 + op["_b"].size * 4

        if op["type"] == "conv":
            meta.append(
                f"  {{ VAS_OP_CONV, {op['cin']}, {op['cout']}, {op['kh']}, {op['kw']}, "
                f"{op['sh']}, {op['sw']}, {op['ph']}, {op['pw']}, {op['groups']}, "
                f"{1 if op['relu'] else 0}, w{i}_q, w{i}_s, b{i} }},"
            )
        else:
            meta.append(
                f"  {{ VAS_OP_LINEAR, {op['cin']}, {op['cout']}, 1, 1, 1, 1, 0, 0, 1, 0, "
                f"w{i}_q, w{i}_s, b{i} }},"
            )
        if op is ops[-1]:
            pass

    # gap sits between the last conv and the linear layer; encode it explicitly
    gap_index = next(i for i, o in enumerate(ops) if o["type"] == "gap")
    meta.insert(gap_index, "  { VAS_OP_GAP, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0 },")

    hdr = f"""// GENERATED by ml/export.py — do not edit by hand.
//
// Sound classifier weights for the ESP32 firmware.
// int16 weights with per-output-channel float scales; biases and activations stay
// float32. Halves the flash of float weights with no calibration pass, and the
// S3's FPU handles the float accumulate comfortably at 4 inferences/sec.
//
// int8 was tried first and rejected: it changed the model's gate decision on ~4.4%
// of real test windows, because the error accumulates across ten layers. int16
// brings that back to noise for 35 kB more flash on an 8 MB part.
//
// Total weight storage: {total_bytes / 1024:.1f} kB
#pragma once
#include <stdint.h>

#define VAS_N_CLASSES {len(ckpt['classes'])}
#define VAS_N_BINS    {DSP_CFG.shape[0]}
#define VAS_N_FRAMES  {DSP_CFG.shape[1]}
#define VAS_NORM_MEAN {ckpt['norm']['mean']:.8f}f
#define VAS_NORM_STD  {ckpt['norm']['std']:.8f}f
#define VAS_TEMPERATURE {ckpt['temperature']:.8f}f
#define VAS_CONF_THRESHOLD {CONF_THRESHOLD:.4f}f

static const char *VAS_CLASSES[VAS_N_CLASSES] = {{{
    ", ".join(f'"{c}"' for c in ckpt["classes"])
}}};

typedef enum {{ VAS_OP_CONV = 0, VAS_OP_GAP = 1, VAS_OP_LINEAR = 2 }} vas_op_kind;

typedef struct {{
  vas_op_kind kind;
  int cin, cout, kh, kw, sh, sw, ph, pw, groups, relu;
  const int16_t *w_q;
  const float  *w_s;   // one scale per output channel
  const float  *b;
}} vas_layer;

{"".join(chunks)}
static const vas_layer VAS_LAYERS[] = {{
{chr(10).join(meta)}
}};
#define VAS_N_LAYERS (sizeof(VAS_LAYERS) / sizeof(VAS_LAYERS[0]))
"""
    C_OUT.parent.mkdir(parents=True, exist_ok=True)
    C_OUT.write_text(hdr)
    return C_OUT


def main() -> None:
    ckpt_path = ARTIFACTS / "model.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path} — run train.py first")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = build()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ops = flatten_graph(model)

    # Folding is easy to get subtly wrong, so check it against the real module.
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(8):
        feat = rng.standard_normal(DSP_CFG.shape).astype(np.float32)
        with torch.no_grad():
            want = model(torch.from_numpy(feat)[None, None]).numpy().ravel()
        got = run_graph(ops, feat)
        worst = max(worst, float(np.max(np.abs(want - got))))
    print(f"bn-folding check: max |Δ| vs torch = {worst:.3e}")
    if worst > 1e-3:
        raise SystemExit("batchnorm folding does not reproduce the model — refusing to export")

    js = emit_js(ops, ckpt)
    c = emit_c(ops, ckpt)

    graph_meta = [{k: v for k, v in o.items() if not k.startswith("_")} for o in ops]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "graph.json").write_text(json.dumps({
        "ops": graph_meta, "classes": ckpt["classes"], "norm": ckpt["norm"],
        "temperature": ckpt["temperature"],
        "runtime": {"conf_threshold": CONF_THRESHOLD, "smooth_window": SMOOTH_WINDOW,
                    "smooth_hits": SMOOTH_HITS, "refractory_s": REFRACTORY_S,
                    "infer_hop_s": INFER_HOP_S},
    }, indent=2))

    print(f"wrote {js.relative_to(REPO)}  ({js.stat().st_size / 1024:.0f} kB)")
    print(f"wrote {c.relative_to(REPO)}  ({c.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
