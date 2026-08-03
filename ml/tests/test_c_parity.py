"""Asserts the firmware's C front-end and network match PyTorch.

The C code is a third independent implementation of the same maths, written
against fixed-size buffers and a sparse mel filterbank. That sparse filterbank is
exactly the sort of thing that works for 39 of 40 bands and silently truncates the
last one, so it gets checked rather than trusted.

Runs on the host — no board needed. Tolerances are looser than the JS test
because the firmware computes in float32 throughout (an ESP32 has no cheap
float64), and the int8 weights introduce a real, expected quantisation error.

    ml/.venv/bin/python -m tests.test_c_parity          (from ml/)
"""
from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export import C_OUT, emit_c, flatten_graph  # noqa: E402
from vas_ml.config import ARTIFACTS, CLASSES, CONF_THRESHOLD, DSP_CFG  # noqa: E402
from vas_ml.features import features, normalize  # noqa: E402
from vas_ml.model import build  # noqa: E402

HERE = Path(__file__).resolve().parent
FW = HERE.parents[1] / "firmware" / "esp32"

# float32 accumulation over the whole front-end, vs float64 in numpy
TOL_FEAT = 2e-3

# int8 weights genuinely change the answer. Bounding the raw probability distance
# turned out to be the wrong assertion: it is dominated by windows where two
# classes are nearly tied, where a tiny logit shift moves probability a lot but
# changes nothing the device does. Measured on 400 real test windows, max |Δp| is
# ~0.08 while the *decision* (argmax, and whether it clears the gate) agrees 98%
# of the time. So the distance is reported but bounded loosely, and the real
# assertion is on decision agreement — which is what the firmware has to preserve.
TOL_PROB = 0.15
MIN_ARGMAX_AGREE = 0.97
MIN_DECISION_AGREE = 0.96


def main() -> int:
    ckpt_path = ARTIFACTS / "model.pt"
    model = build()
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        ckpt = ck
        print(f"using trained checkpoint {ckpt_path.name}")
    else:
        torch.manual_seed(4242)
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.normal_(0, 0.5)
                m.running_var.uniform_(0.3, 2.0)
                m.weight.data.uniform_(0.5, 1.5)
                m.bias.data.normal_(0, 0.3)
        ckpt = {"classes": CLASSES, "norm": {"mean": -6.5, "std": 4.0}, "temperature": 1.4}
        print("no checkpoint — testing with randomly initialised weights")
    model.eval()

    ops = flatten_graph(model)
    backup = C_OUT.with_suffix(".h.bak") if C_OUT.exists() else None
    if backup:
        shutil.copy2(C_OUT, backup)

    try:
        emit_c(ops, ckpt)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            exe = td / "c_harness"
            cc = subprocess.run(
                ["cc", "-O2", "-std=c11", "-Wall", "-Wextra", "-Wno-unused-parameter",
                 f"-I{FW}", str(HERE / "c_harness.c"), str(FW / "vas_dsp.c"),
                 str(FW / "vas_nn.c"), "-lm", "-o", str(exe)],
                capture_output=True, text=True,
            )
            if cc.returncode != 0:
                print("compile failed:\n" + cc.stderr)
                return 1
            if cc.stderr.strip():
                print("compiler warnings:\n" + cc.stderr.strip())

            sr, n = DSP_CFG.sample_rate, DSP_CFG.n_samples
            t = np.arange(n) / sr
            rng = np.random.default_rng(31337)
            sigs = [
                np.zeros(n),
                np.sin(2 * np.pi * 440 * t),
                np.sin(2 * np.pi * 3150 * t) * 0.8,
                0.6 * np.sin(2 * np.pi * 660 * t) + 0.4 * np.sin(2 * np.pi * 1320 * t),
                rng.standard_normal(n) * 0.1,
                np.clip(rng.standard_normal(n) * 3, -1, 1),
                np.sin(2 * np.pi * 7700 * t) * 0.5,      # near fmax, tests the top band
                np.sin(2 * np.pi * 25 * t) * 0.5,        # near fmin, tests the bottom band
            ]
            n_synth = len(sigs)

            # Real audio too. Synthetic tones exercise the filterbank edges, but
            # they are not where the model actually operates, and the int8 error
            # measured on them understates the error on real windows (0.03 vs 0.08).
            try:
                from vas_ml.datasets import load_split
                xs, _ = load_split("test")
                pick = rng.choice(len(xs), size=min(150, len(xs)), replace=False)
                sigs += [xs[i].astype(np.float64) / 32768.0 for i in pick]
                print(f"{n_synth} synthetic signals + {len(sigs) - n_synth} real test windows")
            except Exception as e:
                print(f"(no window cache, synthetic signals only: {e})")

            fin, fout = td / "in.bin", td / "out.bin"
            with open(fin, "wb") as f:
                f.write(struct.pack("<i", len(sigs)))
                for s in sigs:
                    f.write(np.asarray(s, dtype="<f4").tobytes())

            run = subprocess.run([str(exe), str(fin), str(fout)],
                                 capture_output=True, text=True)
            print(run.stderr.strip())
            if run.returncode != 0:
                print(f"harness failed ({run.returncode})")
                return 1

            raw = np.fromfile(fout, dtype="<f4")
    finally:
        if backup:
            shutil.move(backup, C_OUT)

    nf = DSP_CFG.shape[0] * DSP_CFG.shape[1]
    stride = nf + 2 * len(CLASSES)
    assert raw.size == stride * len(sigs), f"got {raw.size} floats, expected {stride * len(sigs)}"

    mean, std = ckpt["norm"]["mean"], ckpt["norm"]["std"]
    T = ckpt["temperature"]
    worst_feat = worst_prob = 0.0
    same_argmax = same_decision = 0

    for i, s in enumerate(sigs):
        block = raw[i * stride: (i + 1) * stride]
        c_feat = block[:nf].reshape(DSP_CFG.shape)
        c_probs = block[nf + len(CLASSES):].astype(np.float64)

        py_feat = normalize(features(s), mean, std).astype(np.float64)
        worst_feat = max(worst_feat, float(np.max(np.abs(py_feat - c_feat))))

        with torch.no_grad():
            lg = model(torch.from_numpy(py_feat.astype(np.float32))[None, None]).numpy().ravel()
        z = lg.astype(np.float64) / T
        z -= z.max()
        e = np.exp(z)
        py_probs = e / e.sum()
        worst_prob = max(worst_prob, float(np.max(np.abs(py_probs - c_probs))))

        # What the firmware must actually preserve: the same class, and the same
        # answer to "does this clear the gate".
        if py_probs.argmax() == c_probs.argmax():
            same_argmax += 1
            if (py_probs.max() >= CONF_THRESHOLD) == (c_probs.max() >= CONF_THRESHOLD):
                same_decision += 1

    n = len(sigs)
    argmax_rate = same_argmax / n
    decision_rate = same_decision / n

    print(f"\n{n} signals, shape {DSP_CFG.shape}")
    print(f"  python vs C   max |Δ feature|      = {worst_feat:.3e}   (tol {TOL_FEAT:g})")
    print(f"  python vs C   max |Δ probability|  = {worst_prob:.3e}   (tol {TOL_PROB:g}, int16 weights)")
    print(f"  python vs C   same class           = {same_argmax}/{n} = {argmax_rate:.3f}"
          f"   (min {MIN_ARGMAX_AGREE})")
    print(f"  python vs C   same gate decision   = {same_decision}/{n} = {decision_rate:.3f}"
          f"   (min {MIN_DECISION_AGREE})")

    ok = (worst_feat <= TOL_FEAT and worst_prob <= TOL_PROB
          and argmax_rate >= MIN_ARGMAX_AGREE and decision_rate >= MIN_DECISION_AGREE)
    print("\n" + ("PASS — firmware preserves the decision within quantisation error."
                  if ok else "FAIL — see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
