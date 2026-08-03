"""Asserts the hand-written JS forward pass matches PyTorch.

cnn.js reimplements convolution by hand. This is what proves it right — and it
runs without a trained model, by exporting randomly-initialised weights, so the
export → JS chain can be validated before spending hours on training.

    ml/.venv/bin/python -m tests.test_inference_parity          (from ml/)

Any existing exported weights are backed up and restored, so running this never
clobbers a trained model.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export import JS_OUT, emit_js, flatten_graph, run_graph  # noqa: E402
from vas_ml.config import ARTIFACTS, CLASSES, DSP_CFG  # noqa: E402
from vas_ml.model import build  # noqa: E402

HERE = Path(__file__).resolve().parent
TOL_LOGIT = 1e-3      # float32 accumulation over ~35k MACs in a different order
TOL_PROB = 1e-4


def main() -> int:
    ckpt_path = ARTIFACTS / "model.pt"
    trained = ckpt_path.exists()

    model = build()
    if trained:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        ckpt = ck
        print(f"using trained checkpoint {ckpt_path.name}")
    else:
        torch.manual_seed(20240)
        # Random init leaves batchnorm at mean 0 / var 1, which would make folding
        # trivially correct and prove nothing. Give the BN layers real statistics.
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.normal_(0, 0.5)
                m.running_var.uniform_(0.3, 2.0)
                m.weight.data.uniform_(0.5, 1.5)
                m.bias.data.normal_(0, 0.3)
        ckpt = {"classes": CLASSES, "norm": {"mean": 0.0, "std": 1.0}, "temperature": 1.7}
        print("no checkpoint — testing with randomly initialised weights")
    model.eval()

    ops = flatten_graph(model)

    backup = None
    if JS_OUT.exists():
        backup = JS_OUT.with_suffix(".js.bak")
        shutil.copy2(JS_OUT, backup)
    try:
        emit_js(ops, ckpt)

        rng = np.random.default_rng(99)
        feats = [rng.standard_normal(DSP_CFG.shape).astype(np.float32) for _ in range(6)]
        # Include the degenerate inputs that expose padding and ReLU bugs.
        feats.append(np.zeros(DSP_CFG.shape, dtype=np.float32))
        feats.append(np.full(DSP_CFG.shape, -8.0, dtype=np.float32))

        with tempfile.TemporaryDirectory() as td:
            fin, fout = Path(td) / "f.json", Path(td) / "o.json"
            fin.write_text(json.dumps([f.ravel().tolist() for f in feats]))
            proc = subprocess.run(
                ["node", str(HERE / "infer_harness.mjs"), str(fin), str(fout)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                print("node harness failed:\n", proc.stdout, proc.stderr)
                return 1
            js = json.loads(fout.read_text())
    finally:
        if backup:
            shutil.move(backup, JS_OUT)
        elif not trained and JS_OUT.exists():
            JS_OUT.unlink()   # don't leave random weights lying around

    if js["classes"] != ckpt["classes"]:
        print(f"FAIL classes: {js['classes']} vs {ckpt['classes']}")
        return 1

    worst_logit = worst_prob = worst_np = 0.0
    T = ckpt["temperature"]
    for i, f in enumerate(feats):
        with torch.no_grad():
            want = model(torch.from_numpy(f)[None, None]).numpy().ravel().astype(np.float64)
        got = np.asarray(js["results"][i]["logits"], dtype=np.float64)
        worst_logit = max(worst_logit, float(np.max(np.abs(want - got))))

        # numpy reference graph too — separates "folding is wrong" from "JS is wrong"
        ref = run_graph(ops, f)
        worst_np = max(worst_np, float(np.max(np.abs(want - ref))))

        z = want / T
        z = z - z.max()
        e = np.exp(z)
        want_p = e / e.sum()
        got_p = np.asarray(js["results"][i]["probs"], dtype=np.float64)
        worst_prob = max(worst_prob, float(np.max(np.abs(want_p - got_p))))

    print(f"{len(feats)} inputs, shape {DSP_CFG.shape} -> {len(CLASSES)} classes")
    print(f"  torch vs numpy-reference  max |Δ logit| = {worst_np:.3e}")
    print(f"  torch vs JS               max |Δ logit| = {worst_logit:.3e}")
    print(f"  torch vs JS               max |Δ prob|  = {worst_prob:.3e}")

    ok = worst_logit <= TOL_LOGIT and worst_prob <= TOL_PROB and worst_np <= TOL_LOGIT
    print("\n" + ("PASS — JS forward pass matches PyTorch." if ok else "FAIL — see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
