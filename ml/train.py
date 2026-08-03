"""Trains the classifier and reports the numbers that actually matter in deployment.

    ml/.venv/bin/python train.py --epochs 40

Argmax accuracy is not the target. The deployed system only ever fires when a class
clears CONF_THRESHOLD, so this reports recall *at that gate* and, more importantly,
the false-alarm rate: how often TV, speech and room tone push an alert class over
0.75. A system that misses a doorbell is annoying; one that cries fire at the
television gets unplugged.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from vas_ml.augment import augment, spec_augment
from vas_ml.config import (
    ALERT_CLASSES, ARTIFACTS, CLASSES, CLASS_IDX, CONF_THRESHOLD, DSP_CFG,
)
from vas_ml.datasets import load_split
from vas_ml.features import features, normalize
from vas_ml.model import build, summarise

BG = CLASS_IDX["background"]


class Windows(Dataset):
    def __init__(self, split: str, train: bool, mean: float = 0.0, std: float = 1.0,
                 seed: int = 0):
        self.x, self.y = load_split(split)
        self.train = train
        self.mean, self.std = mean, std
        self.seed = seed
        self.epoch = 0
        # Real room tone to mix under training examples, drawn from this split only
        # so no validation audio leaks in through the noise channel.
        self.noise_idx = np.flatnonzero(self.y == BG)

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        # Seeded per (epoch, index) so runs are reproducible but every epoch sees
        # a different augmentation of the same window.
        rng = np.random.default_rng((self.seed, self.epoch, i))
        x = self.x[i].astype(np.float64) / 32768.0

        if self.train:
            noise = None
            if self.noise_idx.size and self.y[i] != BG:
                j = self.noise_idx[rng.integers(len(self.noise_idx))]
                noise = self.x[j].astype(np.float64) / 32768.0
            x = augment(x, rng, noise)

        f = features(x).astype(np.float32)
        if self.train:
            f = spec_augment(f, rng)
        f = normalize(f, self.mean, self.std)
        return torch.from_numpy(f[None]), int(self.y[i])


def norm_stats(split: str = "train", n: int = 3000) -> tuple[float, float]:
    """Global scalar mean/std over un-augmented training features."""
    x, _ = load_split(split)
    rng = np.random.default_rng(7)
    idx = rng.choice(len(x), size=min(n, len(x)), replace=False)
    acc = [features(x[i].astype(np.float64) / 32768.0) for i in idx]
    a = np.stack(acc)
    return float(a.mean()), float(a.std() + 1e-8)


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    L, Y = [], []
    for xb, yb in loader:
        L.append(model(xb.to(device)).float().cpu().numpy())
        Y.append(yb.numpy())
    return np.concatenate(L), np.concatenate(Y)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling on the validation set.

    The 0.75 gate is only meaningful if the probabilities mean something. Networks
    trained with cross-entropy are systematically overconfident, so we fit a single
    scalar T that minimises validation NLL and divide the logits by it at inference.
    """
    lg = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    lossf = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = lossf(lg / logT.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp().item())


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


ALERT_I = [CLASS_IDX[c] for c in ALERT_CLASSES]


def recall_at_fixed_fa(logits: np.ndarray, y: np.ndarray, fa_target: float = 0.005) -> float:
    """Mean alert-class recall at the threshold that pins the tv_music/background
    false-alarm rate to `fa_target`.

    This replaces plain argmax alert-recall as the checkpoint-selection metric, and
    the reason is worth recording. Argmax alert-recall has no false-positive term:
    a model that ignores the microphone and cycles through the four alert classes
    scores 0.281 on it — better than the first two epochs of a real run. Selecting
    on it saved this model at epoch 8 of 40, at its most trigger-happy, and threw
    away 32 epochs of genuine improvement.

    Refitting the threshold every epoch makes this threshold- and calibration-free
    by construction: temperature scaling or a change to CONF_THRESHOLD slides a
    model *along* its own curve and cannot change this number. So it ranks curves,
    not operating points. A model that never fires scores 0; one that fires on
    everything pays for it in the threshold this is forced to pick.
    """
    p = softmax(logits)
    pred, conf = p.argmax(1), p.max(1)
    non = np.isin(y, [CLASS_IDX["tv_music"], BG])
    if not non.any():
        return 0.0

    fa_conf = np.sort(conf[non & np.isin(pred, ALERT_I)])[::-1]
    budget = int(fa_target * non.sum())          # this many false alarms are allowed
    thr = float(fa_conf[budget]) if budget < len(fa_conf) else 0.0

    return float(np.mean([
        ((pred == i) & (conf > thr) & (y == i)).sum() / max(1, (y == i).sum())
        for i in ALERT_I
    ]))


def report(logits: np.ndarray, y: np.ndarray, temperature: float, title: str) -> dict:
    p = softmax(logits / temperature)
    pred = p.argmax(1)
    acc = float((pred == y).mean())

    print(f"\n=== {title} ===")
    print(f"argmax accuracy {acc:.3f}   (n={len(y)})")

    # Confusion matrix
    K = len(CLASSES)
    cm = np.zeros((K, K), dtype=int)
    for t, q in zip(y, pred):
        cm[t, q] += 1
    w = max(len(c) for c in CLASSES)
    print(f"\n{'true \\ pred':>{w}} " + " ".join(f"{c[:7]:>7}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>{w}} " + " ".join(f"{v:7d}" for v in cm[i]))

    # What the deployed system actually does: fire only above the gate.
    print(f"\nat the {CONF_THRESHOLD:.0%} confidence gate:")
    print(f"{'class':>{w}}  {'recall':>7} {'precis':>7} {'n':>6}")
    per_class = {}
    for i, c in enumerate(CLASSES):
        fires = (pred == i) & (p.max(1) >= CONF_THRESHOLD)
        tp = int((fires & (y == i)).sum())
        fp = int((fires & (y != i)).sum())
        n = int((y == i).sum())
        rec = tp / n if n else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        per_class[c] = {"recall": rec, "precision": prec, "n": n}
        print(f"{c:>{w}}  {rec:7.3f} {prec:7.3f} {n:6d}")

    # The number that decides whether this is livable: non-events that fire an alert.
    non_event = np.isin(y, [CLASS_IDX["tv_music"], BG])
    fired_alert = np.isin(pred, [CLASS_IDX[c] for c in ALERT_CLASSES]) & (p.max(1) >= CONF_THRESHOLD)
    fa = int((non_event & fired_alert).sum())
    n_non = int(non_event.sum())
    rate = fa / n_non if n_non else 0.0
    # Inference runs on a sliding window 4x/second in the browser.
    print(f"\nfalse alarms on tv_music/background: {fa}/{n_non} windows = {rate:.4f}")
    print(f"  ≈ {rate * 4 * 3600:.1f} raw triggers/hour — this is a per-window UPPER BOUND.")
    print("  Run eval_deployed.py for the rate the device actually produces. Do not")
    print("  assume smoothing rescues it: measured, 3-of-5 bought ~12x, not the ~1000x")
    print("  independent draws would predict (consecutive windows overlap 75%, and")
    print("  television is sustained, so the errors are correlated).")

    return {"accuracy": acc, "per_class": per_class,
            "false_alarm_rate": rate, "confusion": cm.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=ARTIFACTS / "model.pt")
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")

    print("computing normalisation stats…")
    mean, std = norm_stats()
    print(f"  mean {mean:.4f}  std {std:.4f}")

    tr = Windows("train", True, mean, std)
    va = Windows("val", False, mean, std)
    te = Windows("test", False, mean, std)
    print(f"windows: train {len(tr)}  val {len(va)}  test {len(te)}")

    dl = dict(batch_size=args.batch, num_workers=args.workers,
              persistent_workers=args.workers > 0, pin_memory=False)
    tr_dl = DataLoader(tr, shuffle=True, drop_last=True, **dl)
    va_dl = DataLoader(va, shuffle=False, **dl)
    te_dl = DataLoader(te, shuffle=False, **dl)

    model = build().to(device)
    print(summarise(model.cpu()))
    model.to(device)

    # Inverse-frequency class weights: doorbell has an order of magnitude fewer
    # windows than music, and it is the class we can least afford to miss.
    #
    # Two guards, both of which cost a full training run to discover the hard way:
    #
    #  - Absent classes get weight 0, not a huge one. Label smoothing puts a little
    #    target mass on *every* class, so a class with no examples still enters the
    #    loss; combined with an inverse-frequency weight of 1/0 that term dominates
    #    and the model learns to predict the class it has never seen.
    #  - The range is clipped. Raw inverse frequency on a 40:1 imbalance weights the
    #    rare class so heavily that the model will trade away everything else for it.
    counts = np.bincount(tr.y, minlength=len(CLASSES)).astype(np.float64)
    present = counts > 0
    weights = np.zeros(len(CLASSES), dtype=np.float64)
    weights[present] = np.clip(
        counts[present].sum() / (present.sum() * counts[present]), 0.25, 8.0)

    absent = [c for c, ok in zip(CLASSES, present) if not ok]
    if absent:
        print(f"!! no training windows for: {', '.join(absent)} — "
              "weighted to zero, and the model cannot ever predict them.")
        print("   (expected if you have only ESC-50; FSD50K supplies doorbell and tv_music)")
    print("class weights " + ", ".join(f"{c}={w:.2f}" for c, w in zip(CLASSES, weights)))
    lossf = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=0.05,   # mild; keeps the model from saturating the 0.75 gate
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(tr_dl), pct_start=0.25)

    best, best_state, best_ep = -1.0, None, 0
    for ep in range(args.epochs):
        tr.set_epoch(ep)
        model.train()
        t0, tot, seen, corr = time.time(), 0.0, 0, 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
            corr += int((out.argmax(1) == yb).sum())
            seen += len(yb)

        vl, vy = collect_logits(model, va_dl, device)
        vacc = float((vl.argmax(1) == vy).mean())
        # Plain argmax alert-recall, kept only so the curve stays visible in the log.
        # It is NOT what selects the checkpoint — see recall_at_fixed_fa.
        alert_rec = np.mean([
            float((vl.argmax(1)[vy == CLASS_IDX[c]] == CLASS_IDX[c]).mean())
            for c in ALERT_CLASSES if (vy == CLASS_IDX[c]).any()
        ])
        score = recall_at_fixed_fa(vl, vy)
        flag = ""
        if score > best:
            best, best_state, best_ep = score, {k: v.detach().cpu().clone()
                                                for k, v in model.state_dict().items()}, ep + 1
            flag = " *"
        print(f"ep {ep+1:3d}/{args.epochs}  loss {tot/seen:.4f}  "
              f"train {corr/seen:.3f}  val {vacc:.3f}  alert-rec {alert_rec:.3f}  "
              f"R@FA {score:.3f}  {time.time()-t0:5.1f}s{flag}")

    model.load_state_dict(best_state)
    print(f"\nselected epoch {best_ep}/{args.epochs}  (R@FA {best:.3f})")
    if best_ep <= args.epochs // 4:
        print("  !! selected very early — check the metric is not rewarding "
              "trigger-happiness again")

    vl, vy = collect_logits(model, va_dl, device)
    T = fit_temperature(vl, vy)
    print(f"\ncalibration temperature T = {T:.3f}")

    report(vl, vy, T, "validation")
    tl, ty = collect_logits(model, te_dl, device)
    test_metrics = report(tl, ty, T, "test (held-out clips)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "classes": CLASSES,
        "norm": {"mean": mean, "std": std},
        "temperature": T,
        "dsp": {"feature": DSP_CFG.feature, "shape": list(DSP_CFG.shape)},
        "test_metrics": test_metrics,
    }, args.out)
    (args.out.parent / "metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
