"""Audio front-end.

Everything above the `--- training-only ---` divider is mirrored line-for-line in
vas3d/js/audio/mel.js and must stay that way: if the browser computes features even
slightly differently from what the model trained on, the model still runs, still
returns confident-looking numbers, and is quietly wrong. tests/test_parity.py is
what stops that happening, so run it after touching anything here.

No librosa on purpose — every formula is written out so it can be reimplemented in
JS and in C without guessing which convention a library picked.
"""
from __future__ import annotations

import numpy as np

from .config import DSP, DSP_CFG


# ---------------------------------------------------------------------------
#  Mel scale (HTK convention — the same formula vas3d/js/config.js already used)
# ---------------------------------------------------------------------------
def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(cfg: DSP = DSP_CFG) -> np.ndarray:
    """(n_mels, n_fft//2 + 1) triangular filters, unit peak (no area normalisation).

    Unit peak rather than Slaney area-normalisation because it is one less
    convention to get wrong in three languages.
    """
    n_out = cfg.n_fft // 2 + 1
    # n_mels + 2 edges: filter m spans edges[m] .. edges[m+2], peaking at edges[m+1]
    mels = np.linspace(hz_to_mel(cfg.fmin), hz_to_mel(cfg.fmax), cfg.n_mels + 2)
    edges_hz = mel_to_hz(mels)
    edges_bin = edges_hz * cfg.n_fft / cfg.sample_rate  # fractional FFT bins

    fb = np.zeros((cfg.n_mels, n_out), dtype=np.float64)
    k = np.arange(n_out, dtype=np.float64)
    for m in range(cfg.n_mels):
        lo, mid, hi = edges_bin[m], edges_bin[m + 1], edges_bin[m + 2]
        if mid > lo:
            up = (k - lo) / (mid - lo)
            fb[m] = np.where((k > lo) & (k <= mid), up, fb[m])
        if hi > mid:
            down = (hi - k) / (hi - mid)
            fb[m] = np.where((k > mid) & (k < hi), down, fb[m])
    return np.clip(fb, 0.0, None)


def hann(n: int) -> np.ndarray:
    """Periodic Hann (scipy's sym=False / numpy's `hanning` shifted by one)."""
    i = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * i / n)


def dct2_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Orthonormal DCT-II, (n_out, n_in)."""
    k = np.arange(n_out, dtype=np.float64)[:, None]
    n = np.arange(n_in, dtype=np.float64)[None, :]
    m = np.cos(np.pi * k * (2.0 * n + 1.0) / (2.0 * n_in))
    m *= np.sqrt(2.0 / n_in)
    m[0] *= np.sqrt(0.5)
    return m


# Built once; both are pure functions of the config.
_FB = mel_filterbank()
_WIN = hann(DSP_CFG.win_length)
_DCT = dct2_matrix(DSP_CFG.n_mels, DSP_CFG.n_mfcc)


# ---------------------------------------------------------------------------
#  The front-end proper
# ---------------------------------------------------------------------------
def frame_signal(x: np.ndarray, cfg: DSP = DSP_CFG) -> np.ndarray:
    """(n_frames, win_length). Input shorter than a clip is zero-padded, longer is cut."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < cfg.n_samples:
        x = np.pad(x, (0, cfg.n_samples - x.shape[0]))
    x = x[: cfg.n_samples]
    idx = np.arange(cfg.win_length)[None, :] + (
        np.arange(cfg.n_frames)[:, None] * cfg.hop_length
    )
    return x[idx]


def power_spectrum(frames: np.ndarray, cfg: DSP = DSP_CFG) -> np.ndarray:
    """(n_frames, n_fft//2+1) magnitude-squared spectrum of windowed, zero-padded frames."""
    win = frames * _WIN[None, :]
    padded = np.zeros((win.shape[0], cfg.n_fft), dtype=np.float64)
    padded[:, : cfg.win_length] = win
    spec = np.fft.rfft(padded, n=cfg.n_fft, axis=1)
    return (spec.real ** 2) + (spec.imag ** 2)


def log_mel(x: np.ndarray, cfg: DSP = DSP_CFG) -> np.ndarray:
    """(n_mels, n_frames) log-mel spectrogram."""
    p = power_spectrum(frame_signal(x, cfg), cfg)     # (frames, bins)
    mel = p @ _FB.T                                    # (frames, n_mels)
    return np.log(mel + cfg.log_floor).T               # (n_mels, frames)


def mfcc(x: np.ndarray, cfg: DSP = DSP_CFG) -> np.ndarray:
    """(n_mfcc, n_frames)."""
    return _DCT @ log_mel(x, cfg)


def features(x: np.ndarray, cfg: DSP = DSP_CFG) -> np.ndarray:
    """(n_bins, n_frames), float32, unnormalised. Normalisation is applied at
    train/infer time with the global stats saved next to the weights."""
    out = mfcc(x, cfg) if cfg.feature == "mfcc" else log_mel(x, cfg)
    return out.astype(np.float32)


def normalize(feat: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Global (scalar) standardisation.

    Deliberately NOT per-example: normalising each window by its own statistics
    rescales near-silence up to full contrast, which turns quiet rooms into
    confident detections. Fixed scalars keep silence looking like silence.
    """
    return (feat - mean) / std


# ---------------------------------------------------------------------------
#  --- training-only --- (no JS mirror needed below this line)
# ---------------------------------------------------------------------------
def load_audio(path, sr: int = DSP_CFG.sample_rate) -> np.ndarray:
    """Mono float64 at `sr`. Uses polyphase resampling; returns empty on unreadable files."""
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    try:
        x, file_sr = sf.read(str(path), dtype="float64", always_2d=True)
    except Exception:
        return np.zeros(0, dtype=np.float64)
    x = x.mean(axis=1)
    if file_sr != sr:
        g = gcd(int(file_sr), int(sr))
        x = resample_poly(x, sr // g, file_sr // g)
    return np.ascontiguousarray(x, dtype=np.float64)
