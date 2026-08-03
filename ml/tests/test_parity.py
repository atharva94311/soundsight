"""Asserts the Python and JS front-ends agree.

A mismatch here is the failure mode that costs days: the model trains fine, the
browser runs fine, and the predictions are quietly garbage because the two mel
spectrograms disagree. Run this after touching either features.py or mel.js.

    ml/.venv/bin/python -m tests.test_parity          (from ml/)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vas_ml.config import DSP_CFG  # noqa: E402
from vas_ml.features import features, log_mel, mfcc  # noqa: E402

HERE = Path(__file__).resolve().parent
TOL = 1e-4          # the stated contract
TIGHT = 1e-9        # what we actually expect from two float64 implementations


def make_signals() -> dict[str, list[float]]:
    """Deterministic probes, generated once in Python so both sides see identical input.

    Chosen to exercise the parts that break: silence (log floor), DC (bin 0),
    Nyquist (last bin), impulses (all bins at once), and tones sitting between
    mel edges where triangular interpolation actually matters.
    """
    sr = DSP_CFG.sample_rate
    n = DSP_CFG.n_samples
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(0xA11CE)

    sig: dict[str, np.ndarray] = {
        "silence": np.zeros(n),
        "dc": np.ones(n) * 0.5,
        "impulse": np.eye(1, n, 0).ravel(),
        "late_impulse": np.eye(1, n, n - 1).ravel(),
        "nyquist": np.cos(np.pi * np.arange(n)),
        "tone_440": np.sin(2 * np.pi * 440 * t),
        "tone_3150": np.sin(2 * np.pi * 3150 * t) * 0.8,      # fire alarm carrier
        "doorbell_2tone": 0.6 * np.sin(2 * np.pi * 660 * t) + 0.4 * np.sin(2 * np.pi * 1320 * t),
        "chirp": np.sin(2 * np.pi * (20 + (7800 - 20) * t / t[-1] / 2) * t),
        "white_noise": rng.standard_normal(n) * 0.1,
        "tiny": rng.standard_normal(n) * 1e-7,                # near the log floor
        "loud_clipped": np.clip(rng.standard_normal(n) * 4, -1, 1),
        "short": rng.standard_normal(n // 3) * 0.3,           # exercises zero-padding
        "long": rng.standard_normal(int(n * 1.7)) * 0.3,      # exercises truncation
    }
    return {k: v.astype(np.float64).tolist() for k, v in sig.items()}


def main() -> int:
    signals = make_signals()

    with tempfile.TemporaryDirectory() as td:
        sig_path = Path(td) / "signals.json"
        out_path = Path(td) / "js.json"
        sig_path.write_text(json.dumps(signals))

        proc = subprocess.run(
            ["node", str(HERE / "parity_harness.mjs"), str(sig_path), str(out_path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print("node harness failed:\n", proc.stdout, proc.stderr)
            return 1
        js = json.loads(out_path.read_text())

    if tuple(js["shape"].values()) != DSP_CFG.shape:
        print(f"FAIL shape: js {js['shape']} vs py {DSP_CFG.shape}")
        return 1

    worst = 0.0
    worst_where = ""
    failures = []

    for name, sig in signals.items():
        x = np.asarray(sig, dtype=np.float64)
        want = {
            "logmel": log_mel(x),
            "mfcc": mfcc(x),
            "features": features(x).astype(np.float64),
        }
        for kind, py in want.items():
            got = np.asarray(js["results"][name][kind], dtype=np.float64).reshape(py.shape)
            diff = float(np.max(np.abs(py - got)))
            scale = max(1.0, float(np.max(np.abs(py))))
            rel = diff / scale
            if rel > worst:
                worst, worst_where = rel, f"{name}/{kind}"
            if diff > TOL:
                failures.append(f"  {name:16s} {kind:9s} max|Δ| = {diff:.3e}")

    print(f"{len(signals)} signals x 3 feature kinds, shape {DSP_CFG.shape} ({DSP_CFG.feature})")
    print(f"worst relative disagreement: {worst:.3e}  ({worst_where})")

    if failures:
        print(f"\nFAIL — {len(failures)} above tolerance {TOL:g}:")
        print("\n".join(failures))
        return 1
    if worst > TIGHT:
        print(f"\nPASS, but looser than expected for float64 (> {TIGHT:g}).")
        print("Tolerance is met; worth a look if you changed an algorithm rather than a constant.")
        return 0
    print(f"\nPASS — Python and JS front-ends agree to {TIGHT:g}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
