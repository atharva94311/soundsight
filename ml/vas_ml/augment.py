"""Waveform augmentation.

This file matters more than the architecture does. The model trains on clean,
close-mic'd, well-recorded public clips and then has to work on a cheap MEMS mic
across a room with a fan running. Augmentation is the only part of the pipeline
that models that gap, so it is aggressive on purpose: level, distance (reverb),
noise, mic colouration and clipping all vary per example.

Everything operates on float64 mono at DSP_CFG.sample_rate, in [-1, 1].
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .config import DSP_CFG

SR = DSP_CFG.sample_rate


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def time_shift(x: np.ndarray, rng, max_s: float = 0.3) -> np.ndarray:
    """Circular shift. The event should not always sit at the same offset — at
    inference the sliding window catches it at an arbitrary phase."""
    n = int(rng.uniform(-max_s, max_s) * SR)
    return np.roll(x, n)


def gain(x: np.ndarray, rng, lo_db: float = -15.0, hi_db: float = 6.0) -> np.ndarray:
    return x * (10.0 ** (rng.uniform(lo_db, hi_db) / 20.0))


def polarity(x: np.ndarray, rng) -> np.ndarray:
    return -x if rng.random() < 0.5 else x


def room_reverb(x: np.ndarray, rng, p: float = 0.5) -> np.ndarray:
    """Synthetic exponentially-decaying-noise impulse response.

    Not a real measured IR, but it is the difference between "sound recorded 20 cm
    from the source" and "sound heard from the other side of a room", which is the
    domain shift that actually matters here.
    """
    if rng.random() > p:
        return x
    rt60 = rng.uniform(0.12, 0.6)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[0] += 1.0                                   # keep the direct path dominant
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-12
    return fftconvolve(x, ir)[: len(x)]


def mic_colour(x: np.ndarray, rng, p: float = 0.6) -> np.ndarray:
    """One-pole high-pass and/or low-pass — a crude stand-in for the response of a
    small MEMS capsule and its port, which is nothing like a studio mic."""
    if rng.random() > p:
        return x
    y = x
    if rng.random() < 0.7:                          # high-pass: roll off rumble
        a = np.exp(-2 * np.pi * rng.uniform(40, 250) / SR)
        hp = np.empty_like(y)
        prev_x = prev_y = 0.0
        for i in range(len(y)):                     # tiny loop; vectorising costs clarity
            hp[i] = a * (prev_y + y[i] - prev_x)
            prev_x, prev_y = y[i], hp[i]
        y = hp
    if rng.random() < 0.7:                          # low-pass: cheap capsule top end
        a = np.exp(-2 * np.pi * rng.uniform(3000, 7500) / SR)
        lp = np.empty_like(y)
        prev = 0.0
        for i in range(len(y)):
            prev = (1 - a) * y[i] + a * prev
            lp[i] = prev
        y = lp
    return y


def mix_noise(x: np.ndarray, noise: np.ndarray, rng,
              snr_lo: float = 0.0, snr_hi: float = 25.0) -> np.ndarray:
    """Mix a real background recording in at a target SNR.

    Real room tone from the background pool rather than synthetic white noise —
    fans, traffic and distant speech are what the mic will actually be hearing
    underneath the event.
    """
    if noise is None or noise.size == 0:
        return x
    if len(noise) < len(x):
        noise = np.pad(noise, (0, len(x) - len(noise)), mode="wrap")
    noise = noise[: len(x)]
    snr = rng.uniform(snr_lo, snr_hi)
    scale = rms(x) / (rms(noise) * (10.0 ** (snr / 20.0)) + 1e-12)
    return x + noise * scale


def soft_clip(x: np.ndarray, rng, p: float = 0.15) -> np.ndarray:
    """Loud events on a cheap preamp clip. Teach the model that is still the event."""
    if rng.random() > p:
        return x
    return np.tanh(x * rng.uniform(1.5, 6.0))


def augment(x: np.ndarray, rng, noise: np.ndarray | None = None) -> np.ndarray:
    """Full chain, ordered the way the physical signal path is ordered:
    source → room → background mixes in → mic → preamp."""
    x = time_shift(x, rng)
    x = room_reverb(x, rng)
    x = mix_noise(x, noise, rng) if noise is not None else x
    x = mic_colour(x, rng)
    x = gain(x, rng)
    x = soft_clip(x, rng)
    x = polarity(x, rng)
    peak = np.max(np.abs(x))
    if peak > 1.0:                                  # keep in range without changing shape
        x = x / peak
    return x


def spec_augment(feat: np.ndarray, rng, p: float = 0.5,
                 max_f: int = 6, max_t: int = 8) -> np.ndarray:
    """Frequency and time masking on the (bins, frames) feature, in place.

    Masks to the feature's own minimum rather than zero: after log-mel, zero is a
    fairly loud value, so masking to 0 would paint bright stripes instead of holes.
    """
    if rng.random() > p:
        return feat
    floor = float(feat.min())
    bins, frames = feat.shape
    f = rng.integers(0, max_f + 1)
    if f:
        f0 = rng.integers(0, max(1, bins - f))
        feat[f0: f0 + f, :] = floor
    t = rng.integers(0, max_t + 1)
    if t:
        t0 = rng.integers(0, max(1, frames - t))
        feat[:, t0: t0 + t] = floor
    return feat
