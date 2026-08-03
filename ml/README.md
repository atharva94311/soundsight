# ml — the real classifier

Trains the sound classifier that the browser twin and the ESP32 firmware both run.
Six classes: **doorbell, fire_alarm, baby_cry, glass_break, tv_music, background**.

`tv_music` and `background` are not padding. A system that alerts on the
television is worse than no system, so the model has to learn what the
television *is*, not merely fail to recognise it.

## One command

```bash
./run_all.sh
```

Fetch → cache → train → export → verify. Resumable at every stage: interrupted
downloads continue, an existing window cache is reused, and re-running after a
code change only redoes what changed.

## The stages, individually

```bash
./fetch_data.sh                          # ~31 GB, resumable
.venv/bin/python -m vas_ml.datasets      # → data/cache/*.npy
.venv/bin/python train.py --epochs 40    # → artifacts/model.pt
.venv/bin/python export.py               # → browser JS + firmware C header
```

Setup, if the venv is missing:

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python torch numpy soundfile scipy tqdm
```

## Data

| class | source |
|---|---|
| doorbell | FSD50K `Doorbell` (144 clips) |
| fire_alarm | FSD50K `Alarm` + ESC-50 `clock_alarm` |
| baby_cry | FSD50K `Crying_and_sobbing` + ESC-50 `crying_baby` |
| glass_break | FSD50K `Glass` + ESC-50 `glass_breaking` |
| tv_music | FSD50K `Speech` / `Music`, minus anything alarm-like |
| background | everything else, from both |

FSD50K is multi-label, so a clip carrying an alert label wins; speech/music is
only used as a TV negative when nothing alarm-like is also tagged. Near-miss
classes (`Siren`, `Bell`, `Buzz`, `Screaming`, …) are **dropped** rather than
used as negatives — teaching the model that a siren is background is how you get
a system that ignores a real fire alarm.

Two known thin spots: doorbell (~144 clips) and baby cry (~151). Augmentation
carries them, but these are the two classes most worth fine-tuning on your own
recordings later.

There is **no pressure-cooker class** — neither dataset has one (`Boiling` and
`Hiss` are the nearest, and neither is a cooker whistle). `cooker` stays a
simulation-only event in the twin.

## Why the pipeline looks like this

**Splits are by clip, never by window.** Two windows from one recording are
near-duplicates; letting them straddle train and validation gives a validation
number that looks excellent and predicts nothing. FSD50K's own train/val split
and ESC-50's folds are used as-is.

**Weak labels are handled by energy.** A 20-second clip tagged `Doorbell` is
mostly not doorbell. Only the loudest few windows per clip are kept for alert
classes; taking every window teaches the model that "doorbell" means "quiet room".

**Windows are cached as int16 audio, not as features**, so augmentation can run
in the waveform domain. This is the part that decides whether a model trained on
clean public clips survives in a real room: level, distance (synthetic reverb),
real background noise mixed at 0–25 dB SNR, one-pole mic colouration, and
clipping all vary per example. See `vas_ml/augment.py`.

**The reported metric is not accuracy.** Accuracy is dominated by the huge
background class and barely moves when the model stops detecting doorbells.
`train.py` reports recall and precision *at the 75% gate the deployed system
actually uses*, plus the false-alarm rate on tv_music and background, which is
the number that decides whether this is livable.

**Confidence is calibrated.** The 75% gate is meaningless if the probabilities
are not, and cross-entropy training is systematically overconfident. A single
temperature is fitted on validation and shipped with the weights.

## Three implementations, one set of numbers

The front-end exists three times — `vas_ml/features.py`, `vas3d/js/audio/mel.js`,
`firmware/esp32/vas_dsp.c` — and the network twice more. If any of them drifts,
the model still runs and still returns confident-looking answers, and is simply
wrong about what it heard. That failure is silent, so it is tested:

```bash
.venv/bin/python -m tests.test_parity             # Python ↔ JS features
.venv/bin/python -m tests.test_inference_parity   # PyTorch ↔ JS network
.venv/bin/python -m tests.test_c_parity           # PyTorch ↔ firmware (compiles the C on the host)
```

Current agreement:

| check | max disagreement |
|---|---|
| Python ↔ JS features | 6.8e-13 |
| PyTorch ↔ JS logits | 1.4e-06 |
| PyTorch ↔ C features | 5.7e-04 (float32 throughout) |
| PyTorch ↔ C probabilities | 1.6e-02 (int8 weights) |

The inference-parity tests run without a trained model by exporting randomly
initialised weights, so the export chain can be validated before spending hours
training.

## Signal geometry

16 kHz mono, 1-second window, 40 ms frames on a 20 ms hop → exactly 49 frames.
40 mel bands, 20–7800 Hz, HTK mel scale, unit-peak triangles, log floor 1e-6.
Standardised with a single global mean/std — deliberately not per-example, since
normalising each window by its own statistics rescales a quiet room up to full
contrast and turns silence into confident detections.

Everything above is defined once in `vas_ml/config.py`, which generates
`vas3d/js/audio/dsp-config.js`. Change it there and re-run
`python -m vas_ml.config`, then retrain.
