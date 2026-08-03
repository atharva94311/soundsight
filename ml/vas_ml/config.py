"""Single source of truth for every number the model and the browser both need.

`emit_js()` writes these same constants into vas3d/js/audio/dsp-config.js, so the
training front-end and the in-browser front-end cannot drift apart. If you change
anything here, re-run `python -m vas_ml.config` and retrain.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ML = REPO / "ml"
DATA = ML / "data"
ARTIFACTS = ML / "artifacts"
JS_OUT = REPO / "vas3d" / "js" / "audio" / "dsp-config.js"


@dataclass(frozen=True)
class DSP:
    # --- capture ---------------------------------------------------------
    sample_rate: int = 16000      # INMP441 I2S feed, and what the twin quotes
    clip_seconds: float = 1.0     # one inference operates on 1 s of audio

    # --- framing ---------------------------------------------------------
    # 40 ms window / 20 ms hop over 1 s lands on exactly 49 frames:
    #   (16000 - 640) / 320 + 1 == 49
    win_length: int = 640
    hop_length: int = 320
    n_fft: int = 1024             # next pow2 above win_length; frames are zero-padded

    # --- mel filterbank --------------------------------------------------
    n_mels: int = 40
    fmin: float = 20.0
    fmax: float = 7800.0
    log_floor: float = 1e-6       # log(mel + floor); keeps silence finite and flat

    # --- feature selection ----------------------------------------------
    # "logmel" (40 bands) trains better than MFCC because convolutions can use the
    # local structure the DCT throws away. "mfcc" (13 coeffs) is here because it is
    # cheaper on the ESP32 and it is what an earlier version of the twin advertised.
    feature: str = "logmel"
    n_mfcc: int = 13

    @property
    def n_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))

    @property
    def n_frames(self) -> int:
        return (self.n_samples - self.win_length) // self.hop_length + 1

    @property
    def n_bins(self) -> int:
        return self.n_mfcc if self.feature == "mfcc" else self.n_mels

    @property
    def shape(self) -> tuple[int, int]:
        """(bins, frames) — the CNN input."""
        return (self.n_bins, self.n_frames)


DSP_CFG = DSP()

# ---------------------------------------------------------------------------
#  Classes
# ---------------------------------------------------------------------------
# Order is the model's output order. `background` last so it reads as the default.
CLASSES = ["doorbell", "fire_alarm", "baby_cry", "glass_break", "tv_music", "background"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

# Which of these ever raise an alert in the twin. tv_music and background are the
# two the model must actively recognise in order to stay quiet.
ALERT_CLASSES = ["doorbell", "fire_alarm", "baby_cry", "glass_break"]

# Maps our classes onto the source datasets.
#   fsd50k: clip is ours if it carries ANY of these FSD50K labels
#   esc50 : clip is ours if its ESC-50 category is in this list
# Resolution when a clip matches more than one class is by CLASSES order
# (an alert label beats tv_music, which beats background) — see datasets.py.
SOURCE_LABELS = {
    "doorbell":    {"fsd50k": ["Doorbell"],                "esc50": []},
    "fire_alarm":  {"fsd50k": ["Alarm"],                   "esc50": ["clock_alarm"]},
    "baby_cry":    {"fsd50k": ["Crying_and_sobbing"],      "esc50": ["crying_baby"]},
    "glass_break": {"fsd50k": ["Glass"],                   "esc50": ["glass_breaking"]},
    "tv_music":    {"fsd50k": ["Speech", "Music"],         "esc50": []},
    # background is "matched nothing above", not an explicit list.
    "background":  {"fsd50k": [],                          "esc50": []},
}

# Labels that veto a class even though that class's own label matched.
#
# FSD50K inherits the AudioSet ontology, which is HIERARCHICAL: a clip is tagged
# with its ancestors too. Every doorbell is tagged `Doorbell,Bell,Alarm`, because
# Alarm is an ancestor of Doorbell. Without this table `Alarm` matched fire_alarm
# on every one of them, the clip looked like two alert classes at once, and all
# 107 doorbell clips in FSD50K were silently discarded as ambiguous — the model
# then trained happily on five classes and reported success.
#
# The same mechanism keeps sirens out of fire_alarm. A siren clip is tagged
# `Siren,Alarm`; 77 of them would otherwise teach the model that a passing
# ambulance is a house fire.
#
# Rule: the more specific label wins, and confusable siblings are excluded.
CLASS_EXCLUDE = {
    "fire_alarm": [
        "Doorbell",        # more specific: belongs to the doorbell class
        "Siren",           # a siren is not a fire alarm
        "Bicycle_bell", "Church_bell", "Cowbell",
        "Telephone",
    ],
}

# FSD50K labels that are too close to an alert class to be safely called background
# or tv_music. A siren is not a fire alarm, but a clip of one is a terrible negative.
AMBIGUOUS_FSD50K = [
    "Siren", "Bell", "Church_bell", "Bicycle_bell", "Cowbell", "Buzz",
    "Beep_and_bleep", "Fire", "Fireworks", "Gunshot_and_gunfire",
    "Child_speech_and_kid_speaking", "Screaming", "Whimper", "Baby_cry_and_infant_cry",
]

# ---------------------------------------------------------------------------
#  Runtime detection policy (browser + firmware share these)
# ---------------------------------------------------------------------------
CONF_THRESHOLD = 0.75    # matches CONF_THRESHOLD in vas3d/js/config.js
SMOOTH_WINDOW = 5        # consecutive inferences averaged before deciding
SMOOTH_HITS = 3          # ...of which this many must clear the threshold
REFRACTORY_S = 8.0       # after firing, ignore the same class this long
INFER_HOP_S = 0.25       # run inference 4x/second on a sliding 1 s window


def emit_js() -> Path:
    """Write the JS mirror of these constants."""
    cfg = asdict(DSP_CFG)
    body = {
        **cfg,
        "n_samples": DSP_CFG.n_samples,
        "n_frames": DSP_CFG.n_frames,
        "n_bins": DSP_CFG.n_bins,
        "classes": CLASSES,
        "alert_classes": ALERT_CLASSES,
        "conf_threshold": CONF_THRESHOLD,
        "smooth_window": SMOOTH_WINDOW,
        "smooth_hits": SMOOTH_HITS,
        "refractory_s": REFRACTORY_S,
        "infer_hop_s": INFER_HOP_S,
    }
    lines = [
        "// GENERATED by ml/vas_ml/config.py — do not edit by hand.",
        "// Run `python -m vas_ml.config` in ml/ to regenerate.",
        "",
        "export const DSP = Object.freeze({",
    ]
    for k, v in body.items():
        lines.append(f"  {k}: {json.dumps(v)},")
    lines += ["});", ""]
    JS_OUT.parent.mkdir(parents=True, exist_ok=True)
    JS_OUT.write_text("\n".join(lines))
    return JS_OUT


if __name__ == "__main__":
    p = emit_js()
    print(f"wrote {p.relative_to(REPO)}")
    print(f"input shape {DSP_CFG.shape}  ({DSP_CFG.feature})")
    print(f"classes      {CLASSES}")
