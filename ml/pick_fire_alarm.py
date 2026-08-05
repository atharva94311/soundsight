"""Finds a long, robustly-detected fire-alarm clip, and builds a looped version.

    ml/.venv/bin/python pick_fire_alarm.py

fire_alarm is the weakest class this model has (FSD50K's `Alarm` label is a
grab-bag: alarm clocks, buzzers, microwave beeps), and the first clip picked was
2.6 s — too short for a decision rule that needs 3 of 5 windows spanning ~2 s.

So this searches every candidate rather than a sample, requires real length, and
ranks by how many times the clip would fire the deployed rule under room noise —
more firings means more chances for it to latch when played out loud.

It also writes a looped version. A real fire alarm is a repeating pattern (ISO
8201 T-3: three 0.5 s bursts, pause, repeat), so looping is acoustically faithful
rather than a cheat, and it gives the smoothing window plenty to work with.
"""
from __future__ import annotations

import shutil

import numpy as np
import soundfile as sf
import torch

from eval_deployed import clip_probs, decide_stream
from pick_demo_clips import OUT, mix, rms
from vas_ml.config import ARTIFACTS, CLASS_IDX, CLASSES, CONF_THRESHOLD, DSP_CFG
from vas_ml.datasets import index_esc50, index_fsd50k
from vas_ml.features import load_audio
from vas_ml.model import build

MIN_SECONDS = 5.0
SNRS = [15.0, 10.0, 6.0]     # 6 dB is a properly noisy room


def main() -> None:
    ckpt = torch.load(ARTIFACTS / "model.pt", map_location="cpu", weights_only=False)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = build().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mu, sd, T = ckpt["norm"]["mean"], ckpt["norm"]["std"], ckpt["temperature"]

    clips = index_fsd50k() + index_esc50()
    rng = np.random.default_rng(17)

    bg = [c for c in clips if CLASSES[c.label] == "background"]
    rng.shuffle(bg)
    noises = [np.clip(load_audio(c.path), -1, 1) for c in bg[:12]]
    noises = [n for n in noises if n.size > 16000]

    idx = CLASS_IDX["fire_alarm"]
    cands = [c for c in clips if c.label == idx]
    print(f"searching all {len(cands)} fire_alarm clips (min {MIN_SECONDS}s)…\n")

    scored = []
    for i, c in enumerate(cands):
        if i % 150 == 0 and i:
            print(f"  …{i}/{len(cands)}", flush=True)
        x = load_audio(c.path)
        dur = x.size / DSP_CFG.sample_rate
        if dur < MIN_SECONDS:
            continue
        x = np.clip(x, -1, 1)

        fires, conds, confs = 0, 0, []
        for variant in [x] + [mix(x, noises[rng.integers(len(noises))], s) for s in SNRS]:
            p = clip_probs(model, variant, mu, sd, T, device)
            if len(p) < 5:
                continue
            hits = [k for _, k in decide_stream(p, CONF_THRESHOLD) if k == idx]
            if hits:
                conds += 1
                fires += len(hits)
            confs.append(float(p[:, idx].max()))
        if confs:
            # conditions survived first, then how often it fires, then confidence
            scored.append((conds, fires, float(np.mean(confs)), dur, c))

    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    print(f"\n{len(scored)} clips over {MIN_SECONDS}s. Top 8:\n")
    print(f"  {'file':18s} {'len':>6s} {'conds':>6s} {'fires':>6s} {'conf':>6s}")
    for conds, fires, conf, dur, c in scored[:8]:
        print(f"  {c.path.name:18s} {dur:5.1f}s {conds:>3d}/4 {fires:>6d} {conf:>6.2f}")

    if not scored:
        raise SystemExit("no fire_alarm clip long enough — lower MIN_SECONDS")

    conds, fires, conf, dur, best = scored[0]
    dst = OUT / "fire_alarm.wav"
    shutil.copy2(best.path, dst)
    print(f"\nwrote {dst.name}  ({best.path.name}, {dur:.1f}s, fires {fires}x, "
          f"survives {conds}/4 noise conditions)")

    # Looped version: a real alarm repeats, so this is faithful and gives the
    # 5-window smoothing many more chances to latch when played through speakers.
    x = np.clip(load_audio(best.path), -1, 1)
    target = 30.0
    reps = max(2, int(np.ceil(target * DSP_CFG.sample_rate / x.size)))
    gap = np.zeros(int(0.35 * DSP_CFG.sample_rate))     # short pause, as T-3 has
    loop = np.concatenate([np.concatenate([x, gap]) for _ in range(reps)])
    loop = loop[: int(target * DSP_CFG.sample_rate)]

    p = clip_probs(model, loop, mu, sd, T, device)
    n_fire = len([k for _, k in decide_stream(p, CONF_THRESHOLD) if k == idx])
    noisy = clip_probs(model, mix(loop, noises[0], 10.0), mu, sd, T, device)
    n_fire_noisy = len([k for _, k in decide_stream(noisy, CONF_THRESHOLD) if k == idx])

    lp = OUT / "fire_alarm_loop.wav"
    sf.write(lp, loop.astype(np.float32), DSP_CFG.sample_rate)
    print(f"wrote {lp.name}  ({len(loop)/DSP_CFG.sample_rate:.0f}s, "
          f"fires {n_fire}x clean / {n_fire_noisy}x at 10 dB SNR)")


if __name__ == "__main__":
    main()
