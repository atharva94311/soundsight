"""Turns FSD50K + ESC-50 into a cache of labelled 1-second audio windows.

Three things here are load-bearing:

1. **Splits are by clip, never by window.** Two windows from the same recording are
   near-duplicates; letting them straddle train/val produces a validation number
   that looks great and means nothing. FSD50K ships its own train/val split and a
   separate eval set, so we use those; ESC-50 is split on its official folds.

2. **FSD50K is weakly labelled.** A 20 s clip tagged `Doorbell` is mostly *not*
   doorbell. For alert classes we keep only the highest-energy windows, which is
   where the labelled event actually is; taking every window would train the model
   that "doorbell" means "quiet room".

3. **Windows are cached as int16 audio, not as features.** Features are computed
   per epoch so augmentation can happen in the waveform domain — mixing real room
   noise in at a range of SNRs is the single biggest factor in whether a model
   trained on clean public clips survives in an actual hostel room.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import (
    ALERT_CLASSES, AMBIGUOUS_FSD50K, CLASS_EXCLUDE, CLASS_IDX, CLASSES, DATA,
    DSP_CFG, SOURCE_LABELS,
)
from .features import load_audio

CACHE = DATA / "cache"

# Windows kept per clip, per class. Alert events are short and sparse, so we take
# only the loudest few; ambient classes are stationary, so a couple of samples is
# plenty and keeps one long recording from dominating the class.
#
# Scarce classes get a bigger budget. FSD50K has 144 doorbell clips against ~14k
# music clips, and at a flat 3 windows/clip that left doorbell with 235 training
# windows against 9000 — a 38:1 imbalance that no amount of loss weighting fixes
# cleanly. Taking more windows per doorbell clip trades some window redundancy for
# a much less lopsided class, which is the better bargain here.
WINDOWS_PER_CLIP = {
    "doorbell": 8,
    "baby_cry": 6,
    "fire_alarm": 3,
    "glass_break": 3,
    "tv_music": 2,
    "background": 2,
}

# Per-class window budget per split. Music alone has ~14k clips in FSD50K; without
# a cap the model would mostly learn to detect music.
MAX_WINDOWS = {"train": 9000, "val": 1200, "test": 1500}

# An alert window quieter than this is almost certainly the silence around the
# event rather than the event. RMS on [-1, 1] audio.
MIN_ALERT_RMS = 0.008


@dataclass
class Clip:
    path: Path
    label: int
    split: str          # train | val | test
    source: str         # fsd50k | esc50
    clip_id: str


# ---------------------------------------------------------------------------
#  Index building
# ---------------------------------------------------------------------------
def _fsd_class(labels: set[str]) -> str | None:
    """Resolve FSD50K's multi-label tags to one of ours, or None to drop the clip.

    CLASS_EXCLUDE does the heavy lifting: FSD50K tags are hierarchical, so a
    clip's ancestor labels come along for the ride and would otherwise make every
    specific sound look ambiguous. See the comment on CLASS_EXCLUDE.
    """
    hits = []
    for c in ALERT_CLASSES:
        if not (set(SOURCE_LABELS[c]["fsd50k"]) & labels):
            continue
        if set(CLASS_EXCLUDE.get(c, ())) & labels:
            continue                     # vetoed by a more specific or confusable label
        hits.append(c)
    if len(hits) > 1:
        return None                      # genuinely ambiguous, e.g. Alarm + Glass
    if hits:
        return hits[0]

    ambiguous = bool(set(AMBIGUOUS_FSD50K) & labels)
    if set(SOURCE_LABELS["tv_music"]["fsd50k"]) & labels:
        # Speech/music is only a clean TV negative if nothing alarm-like is in it.
        return None if ambiguous else "tv_music"
    # Everything else is background — unless it is one of the near-miss classes,
    # which we drop rather than teach the model to ignore.
    return None if ambiguous else "background"


def index_fsd50k() -> list[Clip]:
    gt = DATA / "FSD50K.ground_truth"
    out: list[Clip] = []
    specs = [
        ("dev.csv", DATA / "FSD50K.dev_audio", None),      # split comes from the csv
        ("eval.csv", DATA / "FSD50K.eval_audio", "test"),
    ]
    for csv_name, audio_dir, forced_split in specs:
        path = gt / csv_name
        if not path.exists():
            continue
        for row in csv.DictReader(open(path)):
            cls = _fsd_class(set(row["labels"].split(",")))
            if cls is None:
                continue
            wav = audio_dir / f"{row['fname']}.wav"
            out.append(Clip(wav, CLASS_IDX[cls], forced_split or row["split"],
                            "fsd50k", row["fname"]))
    return out


def index_esc50() -> list[Clip]:
    root = DATA / "ESC-50-master"
    meta = root / "meta" / "esc50.csv"
    if not meta.exists():
        return []
    # Official folds: 1-3 train, 4 val, 5 test — keeps ESC-50's own cross-validation
    # discipline instead of shuffling clips that share a source recording.
    fold_split = {1: "train", 2: "train", 3: "train", 4: "val", 5: "test"}
    by_cat = {cat: c for c, s in SOURCE_LABELS.items() for cat in s["esc50"]}

    out: list[Clip] = []
    for row in csv.DictReader(open(meta)):
        cls = by_cat.get(row["category"], "background")
        out.append(Clip(root / "audio" / row["filename"], CLASS_IDX[cls],
                        fold_split[int(row["fold"])], "esc50", row["filename"]))
    return out


# ---------------------------------------------------------------------------
#  Window extraction
# ---------------------------------------------------------------------------
def _windows(x: np.ndarray, label: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Pick the windows worth keeping from one clip."""
    n = DSP_CFG.n_samples
    is_alert = CLASSES[label] in ALERT_CLASSES
    want = WINDOWS_PER_CLIP[CLASSES[label]]

    if len(x) < n:
        return [np.pad(x, (0, n - len(x)))] if not is_alert or _rms(x) >= MIN_ALERT_RMS else []

    # Half-window hop gives good coverage without near-identical neighbours.
    starts = list(range(0, len(x) - n + 1, n // 2)) or [0]
    cands = [x[s: s + n] for s in starts]

    if is_alert:
        # The label refers to an event somewhere in the clip; energy is our best
        # available guess at where. Loudest-first, and drop anything near silent.
        cands.sort(key=_rms, reverse=True)
        cands = [w for w in cands if _rms(w) >= MIN_ALERT_RMS]
    else:
        rng.shuffle(cands)
    return cands[:want]


def _rms(w: np.ndarray) -> float:
    return float(np.sqrt(np.mean(w.astype(np.float64) ** 2))) if w.size else 0.0


def source_fingerprint() -> dict:
    """What the cache was built from — both the audio on disk and the rules used.

    Recorded in the cache so a stale cache rebuilds itself. Two ways to go stale:

    - the audio changed (ESC-50 only, then FSD50K arrives). Without this, training
      silently continues on four classes while reporting success.
    - the *labelling rules* changed. File counts alone would not notice, so the
      class map is hashed in too. This matters: the fix that recovered the 107
      dropped doorbell clips changed no files at all.
    """
    fp = {}
    for name in ("FSD50K.dev_audio", "FSD50K.eval_audio", "ESC-50-master/audio"):
        d = DATA / name
        fp[name] = len(list(d.glob("*.wav"))) if d.is_dir() else 0

    rules = json.dumps({
        "classes": CLASSES,
        "source_labels": SOURCE_LABELS,
        "class_exclude": CLASS_EXCLUDE,
        "ambiguous": AMBIGUOUS_FSD50K,
        "windows_per_clip": WINDOWS_PER_CLIP,
        "max_windows": MAX_WINDOWS,
        "min_alert_rms": MIN_ALERT_RMS,
        "n_samples": DSP_CFG.n_samples,
    }, sort_keys=True)
    fp["rules"] = hashlib.sha256(rules.encode()).hexdigest()[:16]
    return fp


def build_cache(force: bool = False) -> dict:
    """Write data/cache/{split}.npy (int16 windows) + meta.json.

    Idempotent, but rebuilds automatically when the available source audio has
    changed since the cache was written.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE / "meta.json"
    fp = source_fingerprint()

    if meta_path.exists() and not force:
        old = json.loads(meta_path.read_text())
        if old.get("sources") == fp:
            print(f"cache present at {CACHE} and sources unchanged "
                  f"(pass force=True to rebuild anyway)")
            return old
        print("source audio has changed since the cache was built — rebuilding")
        print(f"  was {old.get('sources')}")
        print(f"  now {fp}")

    clips = index_fsd50k() + index_esc50()
    if not clips:
        raise SystemExit(
            "No clips indexed. Has ml/fetch_data.sh finished extracting into ml/data?"
        )

    rng = np.random.default_rng(1234)
    # Shuffle so per-class caps sample across the whole set rather than the first
    # N clips (which in FSD50K are correlated by upload batch).
    rng.shuffle(clips)

    buckets: dict[str, dict[int, list[np.ndarray]]] = {
        s: {i: [] for i in range(len(CLASSES))} for s in ("train", "val", "test")
    }
    kept_clips: dict[str, set[str]] = {s: set() for s in buckets}
    missing = 0

    # Progress bar only when attached to a terminal — in a background log the
    # carriage returns produce thousands of unreadable lines.
    import sys as _sys
    for clip in tqdm(clips, desc="windowing", unit="clip",
                     disable=not _sys.stderr.isatty(), mininterval=5.0):
        if clip.split not in buckets:
            continue
        bucket = buckets[clip.split][clip.label]
        if len(bucket) >= MAX_WINDOWS[clip.split]:
            continue
        if not clip.path.exists():
            missing += 1
            continue
        x = load_audio(clip.path)
        if x.size == 0:
            missing += 1
            continue
        # Guard against clipping-on-cast without changing relative levels.
        x = np.clip(x, -1.0, 1.0)
        for w in _windows(x, clip.label, rng):
            bucket.append((w * 32767.0).astype(np.int16))
            kept_clips[clip.split].add(clip.clip_id)

    summary = {"classes": CLASSES, "splits": {}, "missing_files": missing,
               "sample_rate": DSP_CFG.sample_rate, "n_samples": DSP_CFG.n_samples,
               "sources": fp}

    for split, per_class in buckets.items():
        xs, ys = [], []
        for label, ws in per_class.items():
            xs.extend(ws)
            ys.extend([label] * len(ws))
        if not xs:
            continue
        X = np.stack(xs).astype(np.int16)
        y = np.asarray(ys, dtype=np.int64)
        order = rng.permutation(len(y))
        np.save(CACHE / f"{split}_x.npy", X[order])
        np.save(CACHE / f"{split}_y.npy", y[order])
        summary["splits"][split] = {
            "windows": int(len(y)),
            "clips": len(kept_clips[split]),
            "per_class": {CLASSES[i]: int((y == i).sum()) for i in range(len(CLASSES))},
        }

    meta_path.write_text(json.dumps(summary, indent=2))
    return summary


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """(N, n_samples) int16 windows and (N,) int64 labels.

    The audio is memory-mapped rather than read in. At the full dataset size it is
    ~1.7 GB, and DataLoader workers on macOS may be spawned rather than forked —
    in which case each one would otherwise load its own private copy.
    """
    x = np.load(CACHE / f"{split}_x.npy", mmap_mode="r")
    y = np.load(CACHE / f"{split}_y.npy")
    return x, y


if __name__ == "__main__":
    import sys
    s = build_cache(force="--force" in sys.argv)
    print(json.dumps(s, indent=2))
