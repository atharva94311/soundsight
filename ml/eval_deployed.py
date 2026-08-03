"""Measures what the device actually does, not what a single window scores.

    ml/.venv/bin/python eval_deployed.py

train.py reports per-window precision/recall at a threshold. The deployed system
does something different: it slides a 1 s window every 0.25 s, averages the last 5
inferences, requires the winning class to clear the gate on the *mean* AND on at
least 3 of those 5 individually, then goes deaf to that class for 8 s. That rule
lives in vas3d/js/audio/listener.js and firmware/esp32/vas_soundsight.ino.

The gap between the two numbers is not small and not predictable from theory. An
i.i.d. model of the smoothing rule says false alarms should essentially vanish —
five independent draws rarely all land high. But consecutive windows overlap by
75%, and television is *sustained*, so the errors are strongly correlated and the
rule collapses back toward the per-window rate. Measured, smoothing bought ~12x,
not the ~1000x independence would predict.

So this evaluates on contiguous source audio at the real hop, replaying the real
decision rule. The window cache cannot be used for it: datasets.py permutes every
split and keeps only 2-3 non-contiguous windows per clip, which destroys exactly
the temporal correlation that makes this measurement meaningful.

The headline output is **alerts per hour on non-events**. For a bedside alerting
device, roughly 1 nuisance alert per day (0.042/hour) is the target; much more
than that and the user unplugs it, which is a total failure of the product.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from vas_ml.config import (
    ALERT_CLASSES, ARTIFACTS, CLASS_IDX, CLASSES, CONF_THRESHOLD, DSP_CFG,
    INFER_HOP_S, REFRACTORY_S, SMOOTH_HITS, SMOOTH_WINDOW,
)
from vas_ml.datasets import index_esc50, index_fsd50k
from vas_ml.features import features, load_audio, normalize
from vas_ml.model import build

BG = CLASS_IDX["background"]
ALERT_I = [CLASS_IDX[c] for c in ALERT_CLASSES]
NON_EVENT = {"tv_music", "background"}


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def decide_stream(probs: np.ndarray, gate: float) -> list[tuple[int, int]]:
    """Replay listener.js `_decide()` over a sequence of per-window probabilities.

    Returns [(window_index, class_index)] for each alert the device would raise.
    Deliberately a line-for-line port, including the refractory bookkeeping — if
    this drifts from listener.js the number it produces is meaningless.
    """
    fired: list[tuple[int, int]] = []
    last_fired: dict[int, float] = {}
    hist: list[np.ndarray] = []

    for i, p in enumerate(probs):
        hist.append(p)
        if len(hist) > SMOOTH_WINDOW:
            hist.pop(0)
        if len(hist) < SMOOTH_WINDOW:
            continue

        mean = np.mean(hist, axis=0)
        hits = np.zeros(len(CLASSES), dtype=int)
        for q in hist:
            top = int(q.argmax())
            if q[top] >= gate:
                hits[top] += 1

        best = int(mean.argmax())
        if best == BG:
            continue
        if mean[best] < gate or hits[best] < SMOOTH_HITS:
            continue

        now = i * INFER_HOP_S
        if best in last_fired and now - last_fired[best] < REFRACTORY_S:
            continue
        last_fired[best] = now
        fired.append((i, best))
    return fired


@torch.no_grad()
def clip_probs(model, x: np.ndarray, mean: float, std: float, T: float,
               device: str, batch: int = 256) -> np.ndarray:
    """Per-window probabilities for one clip, at the real inference hop."""
    n, hop = DSP_CFG.n_samples, int(INFER_HOP_S * DSP_CFG.sample_rate)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    starts = range(0, len(x) - n + 1, hop)

    feats = [normalize(features(x[s:s + n]), mean, std) for s in starts]
    if not feats:
        return np.zeros((0, len(CLASSES)), dtype=np.float64)

    out = []
    arr = np.stack(feats)[:, None]
    for i in range(0, len(arr), batch):
        t = torch.from_numpy(arr[i:i + batch]).to(device)
        out.append(model(t).float().cpu().numpy())
    return softmax(np.concatenate(out) / T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", type=float, nargs="+",
                    default=[CONF_THRESHOLD, 0.85, 0.90, 0.95])
    ap.add_argument("--limit", type=int, default=400,
                    help="clips per group; the full test split takes a while")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    ckpt = torch.load(ARTIFACTS / "model.pt", map_location="cpu", weights_only=False)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = build().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean, std = ckpt["norm"]["mean"], ckpt["norm"]["std"]
    T = ckpt["temperature"]

    rng = np.random.default_rng(11)
    clips = [c for c in index_fsd50k() + index_esc50() if c.split == args.split]
    rng.shuffle(clips)

    neg = [c for c in clips if CLASSES[c.label] in NON_EVENT][: args.limit]
    pos = [c for c in clips if CLASSES[c.label] in ALERT_CLASSES][: args.limit]
    print(f"device {device}   negatives {len(neg)} clips   positives {len(pos)} clips")

    # Cache probabilities once, then replay the decision rule at each gate.
    def collect(group):
        out = []
        for i, c in enumerate(group):
            if i % 100 == 0:
                print(f"  ...{i}/{len(group)}", flush=True)
            x = load_audio(c.path)
            if x.size == 0:
                continue
            p = clip_probs(model, np.clip(x, -1, 1), mean, std, T, device)
            if len(p):
                out.append((c, p, len(x) / DSP_CFG.sample_rate))
        return out

    print("scoring negatives…")
    neg_p = collect(neg)
    print("scoring positives…")
    pos_p = collect(pos)

    total_hours = sum(d for _, _, d in neg_p) / 3600.0
    print(f"\nnon-event audio: {total_hours * 60:.1f} min\n")

    hdr = f"{'gate':>6} {'alerts/hr':>10} {'per day':>9} {'detect':>8}   per-class detection"
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for gate in args.gates:
        n_alerts = sum(len(decide_stream(p, gate)) for _, p, _ in neg_p)
        per_hour = n_alerts / total_hours if total_hours else 0.0

        det = {}
        for c in ALERT_CLASSES:
            idx = CLASS_IDX[c]
            grp = [(cl, p) for cl, p, _ in pos_p if cl.label == idx]
            if not grp:
                continue
            hit = sum(1 for _, p in grp
                      if any(k == idx for _, k in decide_stream(p, gate)))
            det[c] = hit / len(grp)
        mean_det = float(np.mean(list(det.values()))) if det else 0.0

        results[f"{gate:.2f}"] = {"alerts_per_hour": per_hour,
                                  "mean_detection": mean_det, "per_class": det}
        print(f"{gate:>6.2f} {per_hour:>10.2f} {per_hour * 24:>9.1f} {mean_det:>8.3f}   "
              + "  ".join(f"{c[:9]} {v:.2f}" for c, v in det.items()))

    print("\ntarget for a bedside device is roughly 1 nuisance alert/day = 0.042/hour")
    (ARTIFACTS / "deployed_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {ARTIFACTS / 'deployed_metrics.json'}")


if __name__ == "__main__":
    main()
