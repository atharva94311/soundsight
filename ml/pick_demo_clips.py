"""Finds the training clip per alert class that this model recognises most robustly.

    ml/.venv/bin/python pick_demo_clips.py

"Easily recognised in a moderately noisy room" is not a judgement to make by ear —
it is a measurement. For each candidate clip this replays the real deployed
decision rule (mean over 5 windows, 3 of 5 above the gate, 8 s refractory) three
times: clean, and with real room noise mixed in at 15 dB and 10 dB SNR. The noise
comes from the background class, so it is actual fans/traffic/room tone rather
than white noise.

Clips are ranked by how much noise they survive first, and confidence second, so
the winner is the one most likely to work when you play it out loud into a laptop
mic with a room around it.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch

from eval_deployed import clip_probs, decide_stream
from vas_ml.config import ALERT_CLASSES, ARTIFACTS, CLASS_IDX, CLASSES, CONF_THRESHOLD, REPO
from vas_ml.datasets import index_esc50, index_fsd50k
from vas_ml.features import load_audio

OUT = REPO / "test-sounds"
SNRS = [15.0, 10.0]          # dB — 15 is a normal room, 10 is a noisy one
CANDIDATES_PER_CLASS = 70


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def mix(x, noise, snr_db):
    if len(noise) < len(x):
        noise = np.pad(noise, (0, len(x) - len(noise)), mode="wrap")
    scale = rms(x) / (rms(noise[: len(x)]) * (10 ** (snr_db / 20.0)) + 1e-12)
    y = x + noise[: len(x)] * scale
    peak = np.max(np.abs(y))
    return y / peak if peak > 1 else y


def main() -> None:
    ckpt = torch.load(ARTIFACTS / "model.pt", map_location="cpu", weights_only=False)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    from vas_ml.model import build
    model = build().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mu, sd, T = ckpt["norm"]["mean"], ckpt["norm"]["std"], ckpt["temperature"]

    clips = index_fsd50k() + index_esc50()
    rng = np.random.default_rng(5)

    # Real room tone to mix under the candidates.
    bg = [c for c in clips if CLASSES[c.label] == "background"]
    rng.shuffle(bg)
    noises = []
    for c in bg[:12]:
        n = load_audio(c.path)
        if n.size > 16000:
            noises.append(np.clip(n, -1, 1))
    print(f"{len(noises)} room-noise beds\n")

    OUT.mkdir(parents=True, exist_ok=True)
    picked = {}

    for cls in ALERT_CLASSES:
        idx = CLASS_IDX[cls]
        cands = [c for c in clips if c.label == idx]
        rng.shuffle(cands)
        cands = cands[:CANDIDATES_PER_CLASS]
        print(f"{cls}: testing {len(cands)} clips…")

        scored = []
        for c in cands:
            x = load_audio(c.path)
            if x.size < 16000:            # too short to survive 5-window smoothing
                continue
            x = np.clip(x, -1, 1)

            survives, confs = 0, []
            for variant in [x] + [mix(x, noises[rng.integers(len(noises))], s) for s in SNRS]:
                p = clip_probs(model, variant, mu, sd, T, device)
                if len(p) < 5:
                    continue
                fired = [k for _, k in decide_stream(p, CONF_THRESHOLD)]
                if idx in fired:
                    survives += 1
                confs.append(float(p[:, idx].max()))

            if confs:
                scored.append((survives, float(np.mean(confs)), c, len(x) / 16000))

        if not scored:
            print(f"  !! nothing usable for {cls}\n")
            continue

        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        survives, conf, clip, dur = scored[0]
        dst = OUT / f"{cls}.wav"
        shutil.copy2(clip.path, dst)
        picked[cls] = (survives, conf, dur, clip.path.name)
        print(f"  -> {clip.path.name}  {dur:.1f}s  fires in {survives}/3 conditions  "
              f"peak conf {conf:.2f}\n")

    print("=" * 64)
    print(f"{'class':14s} {'file':16s} {'len':>6s} {'clean':>6s} {'15dB':>6s} {'10dB':>6s}")
    for cls, (s, conf, dur, name) in picked.items():
        marks = ["OK" if s > i else "--" for i in range(3)]
        print(f"{cls:14s} {name:16s} {dur:5.1f}s {marks[0]:>6s} {marks[1]:>6s} {marks[2]:>6s}")
    print(f"\nwrote {len(picked)} clips to {OUT}")


if __name__ == "__main__":
    main()
