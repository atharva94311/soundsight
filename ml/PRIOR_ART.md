# Prior art: tiny sound-event detection for the six classes

`doorbell · fire_alarm · baby_cry · glass_break · tv_music · background`
ESP32-S3 / INMP441 / 16 kHz / (40, 49) log-mel / hand-written int8 C runtime / ~35 kB flash, 130 kB fp32 activations.

**Evidence legend — this matters, do not ignore it.**

| Mark | Meaning |
|---|---|
| **[V]** | Fetched *and* adversarially re-verified: someone tried to break the claim, read the committed config files, downloaded the artifact, and re-derived the numbers. Five items only. |
| **[F]** | Fetched by a finder agent and its page read, but **not** independently re-verified. Numbers and licenses are quoted from what that agent saw. Treat as a strong lead, not as fact. Verify before you build on it. |
| **[S]** | Only seen in search-result text. Nobody opened the page. Verify everything. |

Nothing below is from memory. If a number is not marked, it is not claimed.

---

## 1. Verdict

**No drop-in exists.** There is no open-source model — at any size — whose class list is doorbell / fire_alarm / baby_cry / glass_break / tv_music / background, and there is no MCU-sized artifact that covers more than two of our six. The closest MCU-scale things we found are a 2 k-param Edge Impulse glass-break toy and ST's 5-class FSD50K head (`Speech, Gunshot_and_gunfire, Crying_and_sobbing, Knock, Glass`) — both smaller and less capable than what we already ship, and both on a front end that is not ours.

**Partly, though: three things get us ~80% of the way**, and none of them is a model we run on the MCU. (1) **AudioSet-ontology teachers** — YAMNet, CED, EfficientAT and PANNs all expose every one of our six concepts as named classes, so the pseudo-labeling and distillation path is fully stocked and mostly permissively licensed. (2) **Frame-level labelers** (PretrainedSED, AudioSet-Strong) which cut 1-second windows aligned to the actual transient — the single most likely fix for `fire_alarm` 0.13 and `glass_break` 0.15. (3) **Bulk negative data** (MUSAN, SINS "Watching TV"/"Absence", DESED domestic negatives) for `tv_music` and `background`.

**Correction, checked against the repo after this report was drafted.** An earlier draft of this section claimed `background` recall of 0.001 meant the classifier was mis-wired. **That is wrong, and it was checked and falsified.** The numbers in `artifacts/metrics.json` are recall *at the 75% gate*, exactly as `train.py:5-8` states — not at argmax. At argmax the confusion matrix reads:

| class | at the 75% gate (what `metrics.json` prints) | at argmax (from `confusion`) |
|---|---|---|
| `background` | 0.001 | **0.571** |
| `tv_music` | 0.015 | **0.397** |
| `glass_break` | 0.147 | **0.555** |
| `fire_alarm` | 0.125 | **0.464** |
| `baby_cry` | 0.388 | **0.672** |
| `doorbell` | 0.258 | **0.423** |

`background` is in fact the **most-predicted class of the six** (1,552 predictions against 1,500 true instances). There is no label-mapping bug to find, and the shipped `temperature` is **0.924** — sharpening, not flattening, so calibration is not hiding a good model either. Argmax accuracy is 0.494 across six classes.

The real diagnosis is duller and points the same way as the rest of this report: **the model is genuinely weak and genuinely uncertain**, so almost nothing clears 0.75. That is a data-and-recipe problem, not a wiring problem — which makes §4 (teachers), §5 (data) and §8 Step 4 (distillation) the actual medicine rather than a detour. The published state of the art on the closest academic task (DCASE 2023 Task 4A domestic SED) is PSDS1 ≈ 0.625 with GPU-scale ensembles against a 0.327 baseline [F]; 0.95 is not the target.

---

## 2. The shortlist

Sorted by expected payoff for *our* failure modes, not by category.

| # | Name | What it is | License | Size | Covers | How we'd use it |
|---|---|---|---|---|---|---|
| 1 | **Apple SoundAnalysis `SNClassifySoundRequest(.version1)`** [F] <br>https://developer.apple.com/documentation/soundanalysis/snclassifysoundrequest/knownclassifications | 303-class on-device audio classifier, already installed on every Mac here. Measured at ~214× real time. | Proprietary; weights not extractable. Using its **output** as labels on our own audio is a legal question nobody answered. | n/a | 5 of 6: `door_bell`, `smoke_detector`, `glass_breaking`, `baby_crying`, `music`+`speech`. **No `television` class** → `tv_music` must be composed. | Free, offline, zero-setup bulk pseudo-labeler. 100 h of audio labeled in ~28 min. Also a hard-negative miner (`music`/`chatter`/`babble` → `tv_music`, everything else → `background`). Scripts already written to `…/scratchpad/dump_classes.swift`, `classify.swift`, `apple_soundanalysis_v1_303_classes.txt`. |
| 2 | **CED — `mispeech/ced-tiny…ced-base`** [F] <br>https://huggingface.co/mispeech/ced-base | AudioSet 527-class transformer tagger. Tiny 5.5 M / Mini 9.6 M / Small 22 M / Base 86 M; 48.1–50.0 mAP AS-2M. | **Weights Apache-2.0** on HF (all four). ⚠️ The training repo `RicherMans/CED` is **GPL-3.0** — do not vendor its code. | 5.5–86 M params; 16 kHz, 64-mel | All 6 via `id2label` | **Best distillation teacher for us specifically**: it accepts **variable-length input**, so we can feed it our real 1-second windows instead of zero-padding to 10 s like AST/PaSST/BEATs force. 16 kHz native. int8 ONNX exports exist for laptop-scale bulk labeling (6.1 MB) at https://huggingface.co/k2-fsa/sherpa-onnx-ced-tiny-audio-tagging-2024-04-19 [F]. |
| 3 | **PretrainedSED** [F] <br>https://github.com/fschmid56/PretrainedSED | Frame-level (AudioSet-Strong, ~0.1 s) sound event detection. BEATs / ATST-Frame / fPaSST / ASiT / M2D + low-complexity `frame_mn06`, `frame_mn10`. | **MIT** | Checkpoints auto-download from GitHub Releases; param counts not stated | 447 AudioSet-Strong classes — includes the doorbell / alarm / glass / infant-cry / TV / speech / music branches. Exact 447-list **unverified**. | The window-alignment fix. Clip-level taggers say "this 10 s clip has a doorbell"; we need to know *which 250 ms*. Use it to auto-cut tight 1 s windows centred on the ding-dong or the shatter. Most likely single fix for `fire_alarm` 0.13 and `glass_break` 0.15. |
| 4 | **MUSAN (OpenSLR SLR17)** [F] <br>https://www.openslr.org/17/ | 109 h of `speech` / `music` / `noise`, already 16 kHz mono WAV. ~660 music files, ~930 noise, ~426 speech in 12 languages. | **CC BY 4.0** — shippable | 11 GB | `tv_music` (music + speech) and `background` (noise) | Highest-volume shippable fix for our two broken negative classes. Zero resampling — drops straight into the 40 ms/20 ms front end. **Convolve with room IRs first**: MUSAN is clean source material, not loudspeaker-in-a-room. |
| 5 | **EfficientAT** [F] <br>https://github.com/fschmid56/EfficientAT | MobileNetV3 AudioSet taggers `mn01…mn40` + `dymn`. mn01 = 612 kB / .298 mAP; mn10 = 19.7 MB / .471; mn40 = 274 MB / .487. | **MIT**, and the weights are in this repo's own GitHub Releases so MIT covers them (unlike PANNs). | 612 kB → 274 MB | All 6 (527-class head) | Three uses: (a) the measured **mAP-vs-size curve** — this is our empirical ceiling, not a guess; (b) `mn10_as_mels_40_mAP_453.pt` (19,696,465 B) is trained on **40 mel bands, our band count**; (c) the release also ships **precomputed PaSST ensemble logits** (`passt_enemble_logits_mAP_495.npy`, ~2.0 GB) so we can do transformer→CNN distillation **without ever loading a transformer**. ⚠️ MobileNetV3 uses squeeze-excite + hard-swish — teacher only, not a student template. |
| 6 | **DCASE 2018 Task 5 / SINS** [F] <br>https://zenodo.org/records/1247102 | ~200 h, 72,984 × 10 s segments, 4 mic arrays in a real combined living-room/kitchen, hand-annotated daily activities. | **CC BY-NC 4.0 — cannot ship weights trained on it** | ~45.8 GB | Explicit **`Watching TV`** class (= our `tv_music`, as it actually reaches a room mic) + **`Absence`** (= our `background`, ~hundreds of hours) | The only corpus that is *actually* a fixed room mic hearing a real TV. Use as a pseudo-label / distillation source or an eval set, not as shipped training data. Upstream continuous recordings: https://github.com/KULeuvenADVISE/SINS_database [S] |
| 7 | **TUT Rare Sound Events 2017 (DCASE 2017 Task 2)** [F] <br>https://dcase.community/challenge2017/task-rare-sound-event-detection | Isolated `babycry` / `glassbreak` / `gunshot` events **plus a mixing synthesizer** plus 30 s backgrounds, with EBR recipes (−6/0/+6 dB). | Zenodo 401395: **"Other (Non-Commercial)"** — research/pretraining only | Dev ~17.5 GB (isolated events only = 639.1 MB), eval 3.6 GB | `baby_cry` + `glass_break` directly; backgrounds → `background` | Uncannily close to our class list — see §5. Reported isolated counts (148 babycry / 139 glassbreak / 187 gunshot) came from an arXiv summary, **not** the Zenodo file listing [S]. Even so it roughly triples our glass-break inventory over ESC-50's 40 clips. |
| 8 | **Edge Impulse public project 237590 — "Doorbell Chimes" (Particle)** [F] <br>https://studio.edgeimpulse.com/public/237590/latest | 6,142 audio samples / 1 h 48 m of doorbell-vs-unknown, recorded at **16 kHz**. | **Apache-2.0** on the project dashboard | 1 h 48 m 25 s | `doorbell` + a catch-all negative | We have ~144 doorbell clips. This is the largest shippable single fix for our thinnest positive class, already at our native rate. ⚠️ **Take the data, not the impulse** — the model is 533 kB peak RAM, 4× over budget. ⚠️ Recorded against one Secrui M520+F55 doorbell, so chime *diversity* may be far below 6 k. |
| 9 | **ST `miniresnetv2` FSD50K backbone** — in the **services** repo, not the one you'd find first **[V]** <br>https://github.com/STMicroelectronics/stm32ai-modelzoo-services → `audio_event_detection/tf/src/models/miniresnetv2/pooled_miniresnetv2_1_stacks_backbone.keras` | Head-free, FSD50K-pretrained residual conv backbone terminating in **GlobalAveragePooling2D → 64-d embedding**. Plain conv (not depthwise). | **Apache-2.0**, verbatim, **no ST-device field-of-use clause** — verified. Clean for ESP32-S3. | 455,662 B `.keras`; ~89,280 backbone params → **~90 kB int8** with a GAP+Dense(6) head (2.6× our 35 kB, inside "headroom but not 10×") | None at the label level (ESC-10 head is discarded) — this is a **weights init**, not a class donor | The only commercially-clean pretrained backbone whose shape nearly matches our runtime. Linear-probe then fine-tune on our six classes instead of training from scratch. ⚠️ Three verified gotchas in §3. |
| 10 | **ANSI S3.41 / NFPA 72 Temporal-3 + Temporal-4 cadence** [F/S] <br>Pattern generator: https://goughlui.com/2024/01/29/project-generate-high-quality-industrial-fire-alarm-sounder-audio-with-python/ | `fire_alarm` is not an open-set audio problem — it is a **legally standardized signal**. T3 = 3 × 0.5 s pulses, 0.5 s gaps, then 1.5 s pause (~4 s cycle). T4 = CO. | Pattern itself unencumbered. The `gentonesv4b.zip` generator states **no license** — reimplement from spec (trivial) or ask the author. | 226 tones across 4 sounder families | `fire_alarm` — essentially the whole class | Two moves, both nearly free: **(a)** synthesize unlimited training data sweeping carrier freq, harmonics, RIR, SNR, **and phase offset within the 1 s window** — at 4 Hz some windows contain 3 beeps and some contain the 1.5 s silence, and if training never saw the silence-heavy windows that alone explains 0.13. **(b)** add a ~5 s ring buffer of per-frame `fire_alarm` scores (20 floats/class at 4 Hz, <1 kB RAM) and autocorrelate against the T3/T4 period, in pure C, no retraining. |

**Also grab, cheap and non-negotiable:** `psds_eval` (**MIT**) — https://github.com/DCASE-REPO/psds_eval [F]. It computes effective false-positives **per hour** and **cross-trigger** rate (our `tv_music` head firing on a real doorbell). Per-class recall is close to meaningless for an always-on listener; a 0.9-recall detector that fires 40×/day is unshippable. Use the DCASE-REPO fork — `github.com/audioanalytic/psds_eval` is 404, the org was deleted after Meta acquired Audio Analytic.

---

## 3. Tier 1 — drop-in / near-drop-in for the MCU

Short version: **nothing here is a drop-in.** All five items in this tier were adversarially verified and all five were demoted. Read the gotchas; they are the point of this section.

### 3.1 ST `miniresnetv2` — the one worth taking **[V]**

- Repo (docs/checkpoints): https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/main/audio_event_detection/miniresnetv2/README.md
- **The file to actually take** (head-free, GAP-terminated, plain git, no LFS): `stm32ai-modelzoo-services` → `audio_event_detection/tf/src/models/miniresnetv2/pooled_miniresnetv2_1_stacks_backbone.keras` (455,662 B, Keras 3.8.0, saved 2025-02-24). Sibling `.h5` exists if Keras-3 zip parsing is a problem.
- License **verified**: `miniresnetv2/ST_pretrainedmodel_public_dataset/LICENSE.md` is verbatim Apache 2.0, copyright STMicroelectronics, **not** SLA0044/SLA0048, **no "ST devices only" clause**. Repo-level GitHub metadata says `NOASSERTION` because the tree is mixed-license — the Apache grant is scoped to the `ST_pretrainedmodel_public_dataset` subfolders.
- ST's published figures (verified): 1-stack = 125 K params, 123.98 KiB weights flash, 59.89 KiB activation RAM, 92.5% int8 ESC-10 clip acc, input `(64, 50, 1)`. 2-stack = 440 K params, 93.75%.
- Verified full graph: `Input(64,50,1) → ZeroPad(3,3) → Conv 7×7 s2 ×64 → ZeroPad(1,1) → MaxPool 3×3 s2 → [BN,ReLU,Conv1×1×64,BN,ReLU,ZeroPad1,Conv1×1×64(shortcut),Conv3×3×64,Add] → [BN,ReLU,Conv1×1×64,BN,ReLU,ZeroPad1,MaxPool1×1 s2(shortcut),Conv3×3×64 s2,Add] → BN → ReLU → GAP`. Six Conv2D, five BN, two Add, two MaxPool, four ZeroPad. Keras HWIO → PyTorch OIHW is `transpose(3,2,0,1)`.

**Three verified gotchas that would have burned us:**

1. **RAM does not fit as-is.** ST's 59.89 KiB is **int8** activations; we run **fp32**. Simulated layer-by-layer: at ST-native 64×50 the fp32 peak is **229.5 KiB**; even after dropping to our 40×49 the peak is **~148.5 KiB** (two live buffers around conv1 ≈ 134.9 KiB). Both over 130 kB. **Cheapest fix: cut conv1 from 64 filters to ~24–32 before training.** conv1 is only 3,200 params so almost no pretraining is discarded, and at 40×49 with 32 filters the peak drops to roughly 62 KiB. Decide this *before* training.
2. **Front end is not "essentially no surgery."** ST's verified config: `n_mels=64, n_fft=1024, window_length=1024, hop_length=320, fmin=20, fmax=7500, norm='slaney', htk=False, power=2.0, to_db=True, patch_length=50, target_rate=16000`. Sample rate ✅, hop 320 = 20 ms ✅, fmin 20 ✅, ~1 s patch ✅. Different: 64 vs 40 mels; 64 ms vs our 40 ms window; 7500 vs 7800; **slaney vs our HTK mel** (a genuinely different warping, so different band edges); and `10·log10` power-dB vs our log with a 1e-6 floor. The slaney/HTK delta is the one that erodes exactly the transfer we are buying.
3. **Op set does not match.** We need to add **element-wise Add** and **MaxPool2D** (one of the two is 1×1/stride-2, i.e. pure decimation). ZeroPad folds into conv padding; BN we already fold. The harder part is not the kernels — a residual makes our linear layer-table walker a **DAG**: each layer entry needs an explicit input-tensor id plus a second live buffer for the skip branch. Three implementations (C, JS, PyTorch) to keep parity.

**Also refuted:** the ESC-10 class list is `['dog','chainsaw','crackling_fire','helicopter','rain','crying_baby','clock_tick','sneezing','rooster','sea_waves']` — **no glass class at all**, and `crackling_fire` is flames, not a smoke-detector beep. Real label overlap with our six is **one** class (`crying_baby`). And the cited ESC-10 checkpoint's head is `Flatten(3584)→Dense(10)` (35,850 params, no GAP) — wrong shape for us; that is why the services-repo `pooled_*` backbone is the right file.

### 3.2 ST `yamnet_e256` (FSD50K) — do not port it **[V]**

https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/main/audio_event_detection/yamnet/README.md · Apache-2.0 (same `audio_event_detection/LICENSE.md`, 3-row table covering `yamnet/`, `miniresnetv1/`, `miniresnetv2/` `ST_pretrainedmodel_public_dataset`).

All published numbers verified: 130 K params, input `(64, 96, 1)`, on B-U585I-IOT02A 109.57 kB activation RAM / 135.91 kB weights flash / 167.1 kB total, int8 clip-level 87.0% FSD50K-without-unknown, **73.9% with-unknown**, 94.9% ESC-10. On-disk `.tflite` is **184,240 B** (Git LFS — a naive `raw.githubusercontent` fetch returns a 131-byte pointer; use `media.githubusercontent.com/media/...` or `git lfs pull`).

**Why it is not the answer:**
- **The class list kills the teacher use case.** Committed `config.yaml` pins `class_names = ['Speech', 'Gunshot_and_gunfire', 'Crying_and_sobbing', 'Knock', 'Glass']`. Five-class transfer head, **not** an AudioSet classifier. It cannot label `fire_alarm`, `doorbell`, `tv_music`, or `background` — four of our five broken classes. `Knock` ≠ doorbell (broadband click vs two-tone chime). Counting `Speech` as a `tv_music` proxy is a stretch: our `tv_music` is television and program audio; FSD50K `Speech` is largely live occupants, whom we do **not** want suppressed the same way. Real coverage is **2 of 6**, not 3.
- **It is strictly dominated by its own upstream.** Google's YAMNet exposes all 521 AudioSet classes (see §4.1) from one unauthenticated download.
- **The 110 kB RAM figure is int8.** Our runtime is fp32 → roughly 4×, ~400+ kB. It does not fit. The input tensor alone goes from 7.84 kB (40×49 fp32) to 24.6 kB (64×96 fp32).
- **"Distilled from 3.2 M to 130 K" is not in the source.** The README says only "a much downsized version," never says distilled, never states whether the 256 variant retains AudioSet pretraining. And there is **no FSD50K Yamnet-1024** — the 3.2 M variant appears only in the ESC-10 tables.
- Front end: `n_mels=64, patch_length=96, window_length=400 (25 ms), hop_length=160 (10 ms), fmin=125, fmax=7500, htk=True, power=1.0, to_db=False`. The 10 ms hop **doubles** front-end FFT count (~100/s vs ~50/s) on a battery part before the net runs, and `fmin=125` discards the 20–125 Hz band we deliberately keep.

**Keep it for two things:** its `config.yaml` names the exact FSD50K/AudioSet-ontology label strings for `Glass` and `Crying_and_sobbing` (lift the subset definitions, ST already did the ontology-node→clip-set work via `./src/preprocessing/dataset_utils/fsd50k/audioset_ontology.json`), and the **87.0 → 73.9 drop from merely adding an unknown class on the same dataset** is a measured price tag for an open-set `background` class. Repo is alive: 717 stars, 132 forks, pushed 2026-04-21.

### 3.3 Edge Impulse #233502 "Glass breaking" — 3.5 minutes of glass, that's all **[V]**

https://studio.edgeimpulse.com/public/233502/latest · License **BSD-3-Clause-Clear**, printed on the project page itself.

Everything the finder quoted is accurate — 1,272 samples / 1 h 03 m 07 s @ 44.1 kHz, int8 201 ms / 24.7 K RAM / 28.9 K flash on Cortex-M4F 80 MHz, 100% validation and 100% test. What collapses is what those numbers mean:

- **94% of it is background.** `/acquisition/training`: Background 960 vs Glass_Breaking 57. `/acquisition/testing`: Background 241 vs Glass_Breaking 14. **71 glass clips ≈ 3.5 minutes** — fewer positives than our already-thin doorbell (144) and baby-cry (151) sets.
- **28.9 kB is not comparable to our 35 kB.** Verified graph: `Reshape(40) → Conv1D(8,k3) → MaxPool1D(2) → Dropout → Conv1D(16,k3) → MaxPool1D(2) → Dropout → Flatten → Dense(2)`. Computed param count **≈ 2,138 ≈ 2 kB int8**. The 28.9 kB is EI's whole TFLite-Micro/EON artifact *including operator kernels we do not use*. Apples-to-apples this is **~17× smaller** and correspondingly weaker than what we ship. And `MaxPool1D` + `Flatten` are outside our op set anyway.
- **Front end is 40×99, not 40×49.** 3,960 features / 40 = 99 frames from a 1,000 ms window = 20 ms frame / 10 ms hop. Window increase 500 ms = **2 inferences/sec**, not 4.
- **100%/100% is statistically empty** — n=14 positives, bulk auto-chopped filenames (`output000000715`) from what looks like a handful of source recordings.
- **Licensing splits and the model side fails.** Data is BSD-3-Clause-Clear (commercial-OK, no patent grant). But https://edgeimpulse.com/pricing marks "External distribution to third parties" with an ✗ on the free Developer plan and gates production deployment (≤1000 units) behind an Enterprise Production Phase subscription. **Never ship anything exported from Edge Impulse on the free tier.** Also: no README, no attribution, no provenance for the audio.
- Requires a free EI account to clone (no approval queue, but not a `wget`).

**Verdict: clone it, pull the 71 glass WAVs, resample 44.1→16 kHz with a proper anti-alias filter, fingerprint-dedupe against our existing ESC-50/FSD50K glass, use as positive seeds and a front-end smoke test. Delete the impulse.** Note that 44.1→16 kHz is a real loss *for this class* — glass shatter carries substantial energy above 8 kHz that our 7800 Hz mel ceiling can never see. A few hours of work for 3.5 minutes of audio. Cheap, worth doing, do not schedule around it.

### 3.4 Arm ML-zoo KWS + ML-KWS-for-MCU + MLPerf Tiny — calibration only, and the calibration is inverted **[V]**

- https://github.com/Arm-Examples/ML-zoo/tree/master/models/keyword_spotting — **Apache-2.0**, confirmed three ways. Archived read-only 2025-07-18; last real commit 2023-03-31. All `.tflite` are Git LFS (131-byte in-tree pointers).
- https://github.com/ARM-software/ML-KWS-for-MCU — **Apache-2.0**, real last commit 2018-09-07, TF1 `tf.contrib.slim`.
- https://github.com/mlcommons/tiny — **Apache-2.0** (the finder's "unknown" was wrong). Alive: 470 stars, pushed 2026-07-13.

**Verified int8 sizes / accuracies across the whole Arm KWS line** (downloaded and sha256-checked for micronet_small: 114,512 B, `0b04ee05…`):

| model | int8 bytes | Speech Commands acc |
|---|---|---|
| ds_cnn_small | 47,616 | 93.11% |
| cnn_small | 75,400 | 90.18% |
| dnn_small | 83,544 | 82.11% |
| micronet_small | 114,512 | 95.30% |
| ds_cnn_medium / large | 186,288 / 503,816 | 93.93 / 94.52 |
| micronet_medium / large | 181,968 / 658,832 | 95.8 / 96.5 |

**Refuted: "canonical sub-40 kB reference points."** Nothing in the Arm KWS line is under 40 kB. The smallest shipped artifact is `ds_cnn_small` at 47.6 kB — **36% above our budget, for a task with a 4× smaller input tensor** (490 values vs our 1,960). This inverts the intended reassurance: we sit *below* Arm's smallest published point while feeding 4× more input, so our first conv is proportionally more expensive than theirs.

**Refuted: "bit-for-bit the same framing."** The **time** axis matches exactly (40 ms window / 20 ms stride / 49 frames — confirmed in `recreate_model.sh`: `--window_size_ms 40 --window_stride_ms 20`, and MicroNet's input is `(1,49,10,1)`). The **feature** axis does not: 10 DCT cepstral coefficients vs our 40 log-mel bands. **And the dangerous part:** `data_preprocessing.py` calls `audio_ops.mfcc` with no frequency override, so TensorFlow defaults apply — `tensorflow/core/ops/audio_ops.cc` lines 157–165: `upper_frequency_limit=4000, lower_frequency_limit=20, filterbank_channel_count=40`. **Arm's KWS front end is band-limited to 4 kHz.** Copying it would actively destroy `glass_break` and `fire_alarm`. If anyone proposes "aligning with the Arm reference front end," say no.

**Op sets.** MicroNet needs `DEPTHWISE_CONV_2D` + `RELU6` — we have neither. DS-CNN needs depthwise. **The only Arm KWS model that already fits our layer table is `cnn_small`** (`CONV_2D, FULLY_CONNECTED, RELU, RESHAPE, SOFTMAX`), recipe `--model_architecture cnn --model_size_info 28 10 4 1 1 30 10 4 2 1 16 128`. MicroNet also ships **no training package at any size** — `tflite_int8` only, no checkpoint, no `train.py`, no architecture spec — so it cannot be fine-tuned or inspected.

**Hello Edge correction worth reading.** The intuitive advice "use CNN-S, DS-CNN needs a depthwise op" is **backwards**. `create_cnn_model` in `models.py` has **no GAP** — it flattens the conv map into a low-rank linear then FC; on our 40-band input that flatten head alone goes to **~261 kB int8** vs our 35 kB budget. `create_ds_cnn_model` ends in `slim.avg_pool2d(net, [t_dim, f_dim]) → squeeze → fully_connected(label_count, activation_fn=None)` — literally our op set, missing only depthwise conv, **and its parameter count is independent of input geometry** because the head is GAP. DS-CNN_S sizing string: `5 64 10 4 2 2 64 3 3 1 1 64 3 3 1 1 64 3 3 1 1 64 3 3 1 1`.

**MLPerf Tiny's real value is not the yardstick.** The "35 kB / 130 kB is a reasonable envelope" argument does **not** hold: MLPerf's input is 49×10, ours is 49×40; run the same DS-CNN on our geometry and the first conv alone emits 25×20×64 = 32,000 fp32 activations = **128 kB**, consuming our entire budget. What actually survives is `benchmark/training/keyword_spotting/get_dataset.py:105-125` — `stft(frame_length=480, hop=320, hann) → abs → 40 mel bins via linear_to_mel_weight_matrix (HTK) → log(mel + 1e-6) → DCT[:10]`. **Three edits** (`upper_edge_hertz=7800`, `--window_size_ms 40`, delete the DCT line) turn it into an independent, benchmark-audited reimplementation of **our exact (40, 49) front end** — a fourth parity oracle alongside PyTorch/JS/C. Given `background` at 0.001, a silent front-end bug is a live hypothesis and this is the cheapest way to falsify it. Also lift its augmentation graph (`get_dataset.py:75-103` + `prepare_background_data()` at 199: random time shift + background mixing at `background_frequency=0.8`, `background_volume_range=0.1`) and read `quantize.py` + `quant_cal_idxs.txt` — a worked example of choosing int8 calibration samples from a **curated index list** rather than "first N clips," which is plausibly relevant to `fire_alarm` 0.13 / `glass_break` 0.15 under per-output-channel scales.

Also in mlcommons/tiny and missed by the first pass: `benchmark/training/streaming_wakeword/` — 40 log-mel, 16 kHz, 3 classes (`target`/`unknown`/`silent`), MatchboxNet-style depthwise-separable 1D conv **with residual adds**, QAT, MUSAN augmentation, weights included. README self-describes as "in progress"; its 64 ms/32 ms framing is not ours.

### 3.5 What else lives in this tier (all **[F]**, none lifted)

| Thing | Why it is here | Why we can't just take it |
|---|---|---|
| `ArmDeveloperEcosystem/ml-audio-classifier-example-for-pico` — https://github.com/ArmDeveloperEcosystem/ml-audio-classifier-example-for-pico | **Apache-2.0**, ~15 K params, `Conv2D(8, 8×8, s2, ReLU) → MaxPool → Flatten → Dropout → Dense(1, sigmoid)`, **fire-alarm binary**, trained from **~10 fire-alarm clips** + ESC-50 + Speech Commands as background | Could **not** confirm a trained `.tflite` or C array is committed — treat as a **reproducible recipe**, not weights. It leans on YAMNet embeddings we cannot run on-device. Its existence is strong evidence our `fire_alarm` 0.13 is a data/augmentation problem, not a capacity problem. Colab: https://colab.research.google.com/github/ArmDeveloperEcosystem/ml-audio-classifier-example-for-pico/blob/main/ml_audio_classifier_example_for_pico.ipynb |
| `kahrendt/microWakeWord` — https://github.com/kahrendt/microWakeWord (models: https://github.com/esphome/micro-wake-word-models) | **Apache-2.0**. Closest thing to our deployment: quantized always-on audio on **ESP32-S3** in production inside ESPHome. 40 features, 30 ms window / 10 ms stride, 16 kHz, <10 ms inference. | Wake words only — zero class transfer. Two real lifts: its **C feature-generation code** (`ESPMicroSpeechFeatures`) as a benchmarked ESP32-S3 log-mel front end to compare cycle cost and parity against; and its **streaming formulation** — do a real inference every stride instead of recomputing a full 1 s window 4×/sec. Direct battery win on an 18650. |
| CP-Mobile / DCASE Task 1 baselines — https://github.com/CPJKU/dcase2025_task1_baseline, https://github.com/CPJKU/dcase2024_task1_baseline | 61,148 params / 29.42 MMACs / 122,296 B fp16 — proof a CNN is good at ~60 k params. Inverted-residual: 1×1 pointwise + depthwise + ReLU → GAP → linear. | **No LICENSE file** at either repo (2024 LICENSE URL is a hard 404). Needs residual-add and (full version) GRN. And 29.4 MMACs × 4/sec ≈ 118 MMACs/s on an ESP32-S3 is aggressive. See `Shao_NEPUMSE` at 16.9 MMACs and `Tan_SNTLNTU` at 10.9 MMACs in the DCASE results tables. |
| `yqcai888/easy_dcase_task1` — https://github.com/yqcai888/easy_dcase_task1 | **Apache-2.0** — the only permissively-licensed low-complexity ASC codebase confirmed. TF-SepNet + BC-ResNet in one harness, **and it publishes fine-tuned BEATs logits in GitHub Releases** so we can run teacher-ensemble distillation without downloading a 90 M teacher. TF-SepNet's 1D time/frequency-separated kernels are **plain convs — zero new ops for our runtime**. | Urban scenes, no class overlap. Architecture + distillation harness only. |

---

## 4. Tier 2 — teachers and pseudo-labelers

All too big for the MCU. All solve the actual problem: thin data and collapsed recall. **Every item in this section is [F] — fetched by a finder, not adversarially re-verified.**

### 4.1 YAMNet — take this for the class-index table, at minimum

https://github.com/tensorflow/models/tree/master/research/audioset/yamnet · weights: `https://storage.googleapis.com/audioset/yamnet.h5` (unauthenticated, no form). 3.7 M weights, 69.2 M multiplies per 960 ms frame, balanced mAP 0.306.

**License is split and matters:** the `tensorflow/models` repo code is Apache-2.0; Google's MediaPipe distribution of the same model states the **model** is **CC BY 4.0** (code samples Apache-2.0). Treat the weights as CC-BY — attribution required if shipped.

**Exact indices, read from the real `yamnet_class_map.csv`** (521-class ordering — six classes were dropped by fairness review, so these do **not** match PANNs/CED's 527-class ordering):

```
  0  /m/09x0r    Speech
 19  /m/0463cq4  Crying, sobbing
 20  /t/dd00002  Baby cry, infant cry
132  /m/04rlf    Music
349  /m/03wwcy   Doorbell
350  /m/07r67yg  Ding-dong
393  /m/01y3hg   Smoke detector, smoke alarm
394  /m/0c3f7m   Fire alarm
435  /m/039jq    Glass
437  /m/07rn7sz  Shatter
464  /m/07pc8lb  Breaking
518  /m/06bz3    Television
```

Our six-class mapping:
```
doorbell    ← max(349 Doorbell, 350 Ding-dong)
fire_alarm  ← max(394 Fire alarm, 393 Smoke detector, 382 Alarm)
baby_cry    ← max(20 Baby cry, 19 Crying/sobbing)
glass_break ← max(435 Glass, 437 Shatter, 464 Breaking)
tv_music    ← max(132 Music, 518 Television)
background  ← 1 − max(all of the above)
```

YAMNet is the **weakest** tagger here by mAP — do **not** make it the distillation teacher. Its value is this table plus the fact that its front end (16 kHz mono, ~1 s window, log-mel) is the closest published relative of ours. Deltas: 64 mels over 125–7500 Hz, 25 ms/10 ms framing, `log(mel + 0.001)`. A direct `.tflite` with input `1×15600` (0.975 s @ 16 kHz — Google's own shipped AudioSet classifier operating on essentially our window, which independently validates the 1 s / 4 Hz design) is at `https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite`, labels at `https://storage.googleapis.com/mediapipe-tasks/audio_classifier/yamnet_label_list.txt`.

### 4.2 The rest of the teacher tier

| Model | URL | License | Size / mAP | Take it for |
|---|---|---|---|---|
| **CED** | https://huggingface.co/mispeech/ced-base (+ `-tiny`, `-mini`, `-small`) | **Weights Apache-2.0**; repo GPL-3.0 | Tiny 5.5 M (48.1 AS-2M) / Mini 9.6 M (49.0) / Small 22 M (49.6) / Base 86 M (**50.0**) | **Primary teacher.** Variable-length input → feed it our real 1 s windows. Internally batch-normalizes the mel spectrogram so no dataset mean/var matching. 16 kHz native. Also stores teacher logits **plus the augmentation params that produced them** (+0.3% disk on AudioSet) and trains the student label-free from logits alone — worth copying. |
| **EfficientAT** | https://github.com/fschmid56/EfficientAT (release `v0.0.1`) | **MIT**, weights in its own Releases | `mn01_as_mAP_298.pt` 612,421 B / .298 · `mn02` 1,432,101 B / .378 · `mn04` 4,066,513 B / .432 (0.983 M params, 0.11 B MACs) · `mn05` 5,878,097 B / .443 (1.43 M) · `mn10` 19,708,753 B / .471 (4.88 M, 0.54 B MACs) · `mn20` 71,938,961 B / .478 · `mn30` 156,742,353 B / .482 · `mn40_as_ext` 274,206,417 B / .487 · `dymn04` 1.97 M/.450, `dymn10` 10.57 M/.477, `dymn20` 40.02 M/.491 | The **mAP-vs-width curve**, measured. **`mn10_as_mels_40_mAP_453.pt` (19,696,465 B, .453) is trained on 40 mel bands — our band count.** ⚠️ SE blocks + hard-swish → teacher only. |
| **EfficientAT precomputed PaSST ensemble logits** | https://github.com/fschmid56/EfficientAT/releases/tag/v0.0.1 → `passt_enemble_logits_mAP_495.npy` | **MIT** | 1,995,952,550 B (~2.0 GB), teacher mAP **.495** | Transformer→CNN distillation with **no transformer, no GPU, no teacher inference cost**. This is the exact recipe that produced mn01_as at .298 mAP from 612 kB. Limitation: covers AudioSet clips only — for our own FSD50K/ESC-50 clips we still need a live teacher (CED). |
| **PretrainedSED** | https://github.com/fschmid56/PretrainedSED | **MIT** | BEATs/ATST-Frame/fPaSST/ASiT/M2D in SSL/Weak/Strong variants + `frame_mn06`, `frame_mn10`; checkpoints auto-download | **Frame-level, ~0.1 s.** The window-alignment fix. `frame_mn06/10` are cheap enough to sweep large corpora. |
| **PANNs** | https://github.com/qiuqiangkong/audioset_tagging_cnn · weights Zenodo 3987831 | **Split: code MIT, weights CC BY 4.0** — attribution attaches | `Cnn14_16k_mAP=0.438.pth`, `Cnn14_mAP=0.431`, `MobileNetV1_mAP=0.389` (23.6 MB), `MobileNetV2_mAP=0.383` (20.8 MB); best system .439 | Two reasons to keep despite CED winning on mAP: `Cnn14_16k` is trained **natively at 16 kHz** (no resample, no band-limit mismatch when labeling our own INMP441 recordings); and **MobileNetV1 is pure depthwise-separable + ReLU + global-pool + linear, no SE, no hard-swish — the one AudioSet model whose op set is already inside our layer table**, so it is the most realistic *architectural* template for our student. |
| **AST** | https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593 | **BSD-3-Clause** | 86,195,721 params, mAP .4593 | Easiest to stand up (`AutoModelForAudioClassification`, no `trust_remote_code`). Use as a **cross-check labeler** — keep only pseudo-labels where CED and AST agree. Not primary: below CED-base and it's a fixed-position-embedding ViT wanting ~10 s padded input. |
| **ConvNeXt-Tiny AudioSet** | https://github.com/topel/audioset-convnext-inf (ckpt Zenodo 8020843) | **MIT** | 28,222,767 params, mAP .471, AUC .973 | Best **representation-matched** teacher: it computes log-mel via `torchlibrosa` following the PANNs pipeline, so teacher and student share a feature philosophy and the distillation gap is purely capacity. MIT beats PANNs' CC-BY. |
| **BEATs** | https://github.com/microsoft/unilm/blob/master/beats/README.md | **MIT** (verified at `unilm/blob/master/LICENSE`, no research-only clause) | **No per-checkpoint mAP or file sizes published.** Downloads are OneDrive (`1drv.ms`) — not scriptable. | Third choice. Its more useful form is **inside PretrainedSED**, which repackages it as a frame-level detector under MIT with auto-downloading checkpoints. |
| **PaSST** | https://github.com/kkoutini/PaSST | **Apache-2.0** | `passt_s_swa_p16_128_ap476` (.476) and six siblings .468–.473; auto-download via `get_model(pretrained=True)` | The upstream of the .495 ensemble logits. Mostly unnecessary — but run it if we want teacher soft-targets on **our own** clips from the exact teacher that produced mn01…mn40. |
| **sherpa-onnx CED ONNX** | https://huggingface.co/k2-fsa/sherpa-onnx-ced-tiny-audio-tagging-2024-04-19 | **unknown** on the card (upstream weights Apache-2.0, conversion by k2-fsa) | ced-tiny `model.onnx` 22,256,662 B; `model.int8.onnx` **6,133,417 B**. All four sizes exist. Ships `class_labels_indices.csv` (14,675 B). | Delivery vehicle for the labeling pass: 6.1 MB int8 under `onnxruntime`, no PyTorch, no GPU. Also an ONNX graph to diff against our layer-table runtime as an intermediate parity target. |
| **E-PANNs** | https://github.com/Arshdeep-Singh-Boparai/E-PANNs (ckpt doi 10.5281/zenodo.7939403) | MIT repo; Zenodo license unverified, PANNs parent is CC-BY | 24 M params (~92 MB) vs PANNs 81 M; **pruned mAP not stated** | Bulk labeling workhorse (36% fewer computations, 70% less memory). **Do not use as a teacher** — unquantified mAP. `pip install epanns_inference`. |
| **Audio-MAE** | https://github.com/facebookresearch/AudioMAE | **CC-BY 4.0** (not NC — commercially usable with attribution) | ViT-B, 47.3 mAP AS-2M | Listed so we can rule it out with the license confirmed. Google Drive checkpoints, 10 s ViT, below CED-base. |
| **HTS-AT** | https://github.com/RetroCirce/HTS-Audio-Transformer | **unknown — LICENSE never fetched** | ~30 M params, .471 mAP | Best accuracy-per-parameter among the transformers, **and it ships a DESED SED checkpoint** — closest published task to ours. Worth a second look *if* the license clears. |
| **CLAP (LAION)** | https://github.com/LAION-AI/CLAP · ckpt https://huggingface.co/laion/clap-htsat-unfused | **Repo CC0 1.0; checkpoint Apache-2.0** | ESC-50 zero-shot 89.25–90.14% | **Open-vocabulary data mining.** Text-query an unlabeled corpus: "a doorbell ringing", "a smoke alarm beeping", "glass shattering", "a baby crying", "television dialogue and background music", "quiet empty room". Rank by similarity, hand-verify top-k. Nothing mined is license-contaminated by the labeler. |
| **CLAP (Microsoft)** | https://github.com/microsoft/CLAP | Repo **MIT**; ⚠️ HF `microsoft/msclap` weights showed **`ms-pl`** in search results — verify separately | 2022 / 2023 / `clapcap` checkpoints | Second opinion. Keep only clips where **both** CLAPs agree above threshold. `clapcap` captions mined clips, which catches errors fast (e.g. "a phone ringing" leaking into `doorbell`). |
| **inaSpeechSegmenter** | https://github.com/ina-foss/inaSpeechSegmenter | **MIT** | `keras_speech_music_cnn.hdf5`, `keras_speech_music_noise_cnn.hdf5`; param count unknown | A ready-made **speech/music/noise** discriminator — i.e. `tv_music` vs `background` — that won MIREX 2018 speech detection. Second opinion against AudioSet's noisy `Television` logit (which fires on TV *content*, not TV-ness). |
| **Silero VAD** | https://github.com/snakers4/silero-vad | **MIT** | ~1–2 MB, ONNX + TorchScript, <1 ms per 30 ms chunk | Too big for the MCU. Offline gate to slice speech-bearing regions out of long TV/radio/ambient recordings and bulk-generate `tv_music` / `background`. |
| **google-research/sound-separation (MixIT / FUSS)** | https://github.com/google-research/sound-separation | **Apache-2.0, code and checkpoints** | unknown | Offline data tool: separate a mined clip into sources so we can isolate a doorbell from the TV behind it, then **re-mix at controlled SNR to build hard `tv_music`-vs-`doorbell` pairs**. Attacks "alerts on the television" directly with no license contamination. |

**Apple SoundAnalysis, expanded** [F]. 303 classes, `SNClassifierIdentifierVersion1` is the only identifier; macOS 12+/iOS 15+. Measured **1060 s of 16 kHz audio in 4.95 s wall clock (~214× real time, single process)** on this machine. Empirically verified to fire correctly: synthetic ANSI T3 @ 3100 Hz → `smoke_detector` **0.994–1.000**; synthetic T4 → 0.98–0.999; synthetic glass break → `glass_breaking` 0.964 on the impact window. Relevant labels present: `door_bell, smoke_detector, glass_breaking, baby_crying, music, speech, silence`, plus confusers `beep, alarm_clock, siren, fire_engine_siren, church_bell, bicycle_bell, telephone_bell_ringing, glass_clink, crying_sobbing, chatter, babble`. **There is no `television` class** — `tv_music` must be assembled from `music` + `speech` + `chatter`. Weights are **not** extractable (embedded in the framework binary; no browsable `.mlmodelc` under `/System/Library`) and certainly not shippable. Whether its output may be used as training labels is a question for counsel — no Apple statement was found either way.

**One design signal from Apple worth arguing about internally**: `windowDuration` **defaults to 3.0 s** (range 0.5–15.0 s), for a 303-class model with no compute pressure. Forcing `windowDuration=1.0` still gave `smoke_detector` 0.99+, so 1 s is not fatal — but our two worst positives, `fire_alarm` and `glass_break`, are exactly the two whose discriminative structure lives on a multi-second timescale. **That is evidence the missing recall is in a temporal aggregation layer on top of the 4 Hz classifier, not in a bigger model.**

---

## 5. Tier 3 — data

This is where the 144-doorbell / 151-baby-cry hole actually gets filled. All **[F]** unless marked.

### 5.1 DCASE rare sound events — yes, it checked out, and it is uncannily close

**TUT Rare Sound Events 2017 (DCASE 2017 Task 2)** — https://dcase.community/challenge2017/task-rare-sound-event-detection

The target classes are **exactly `babycry`, `glassbreak`, `gunshot`**. Two of the three are two of our four alert classes, and the third is simply unused. It ships:
- **isolated source events** (639.1 MB) — not 10 s weakly-labeled clips, but the events themselves, which is what we need because we classify 1 s windows and must control onset placement;
- **a mixing synthesizer** with event-to-background-ratio recipes at **−6 / 0 / +6 dB**;
- **30 s backgrounds** drawn from TUT Acoustic Scenes 2016 (15 scenes, domestic and urban) — a principled generator for the `background` negatives we are at 0.001 on.

Sizes: dev ~17.5 GB total (mixtures ~9.2 GB, backgrounds+CV setup ~7.3 GB, isolated events 639.1 MB), eval 3.6 GB. Reported isolated counts — 148 babycry (mean 2.25 s), 139 glassbreak (mean 1.16 s), 187 gunshot (mean 1.32 s) — came from an **arXiv summary, not the Zenodo file listing** [S]; verify.

⚠️ **License: Zenodo record 401395 states "Other (Non-Commercial)"**, terms in `LICENSE.txt` inside the archive. Blocker for a shipped product. Use it for pretraining, pseudo-label research, and evaluation — or contact the authors.

**Download only `isolated_events` + the synthesizer.** That roughly triples our glass-break inventory over ESC-50's 40 clips, and the synthesizer is the artifact to copy regardless of licensing: *isolated event + real domestic background + controlled SNR = strongly-labeled clip*, which simultaneously manufactures the `background` class for free. The same recipe is implemented in DESED via Scaper.

### 5.2 Filling `doorbell` (we have ~144)

| Source | URL | License | Size | Note |
|---|---|---|---|---|
| **Edge Impulse #237590 "Doorbell Chimes"** | https://studio.edgeimpulse.com/public/237590/latest | **Apache-2.0** | 6,142 samples / 1 h 48 m 25 s, **16 kHz** | Largest shippable single fix. Take the data, not the impulse (533 kB peak RAM). One doorbell model (Secrui M520+F55) → check real chime diversity. |
| **AudioSet `Doorbell` + `Ding-dong`** | https://github.com/MorenoLaQuatra/audioset-download | Labels CC-BY-4.0; **audio is YouTube — scraping has ToS implications for a commercial product** | 1.8 M weak clips | `Downloader(labels=["Doorbell","Ding-dong"])` is a two-line call. Pair with AudioSet-Strong for onset-accurate cuts. |
| **agkphysics/AudioSet** (pre-scraped, real audio) | https://huggingface.co/datasets/agkphysics/AudioSet | **CC-BY-4.0** | 2.44 TB FLAC; balanced train 18,683 / balanced test 17,141 / unbalanced train 1,738,657 | Avoids `youtube-dl` against dead IDs. ⚠️ Incomplete (only 18,683 of 22,160 balanced-train; ~303,000 unbalanced-train missing) and **48 kHz 24-bit** — resample to 16 kHz with a proper anti-alias filter or we train on aliasing that does not exist on the INMP441. |
| **Zenodo 3689288 — open-set / few-shot domestic audio events** | https://zenodo.org/records/3689288 | **CC-BY-4.0 — shippable** | 142.1 MB; 1360 clips / 34 classes, 40 per class, 4 s, **mono 16 kHz 16-bit** | **24 distinct "pattern sound" alarm-type classes**, with the abstract naming door bells and fire alarms as the examples. That intra-class *variety* is exactly what a 144-clip doorbell set lacks — the reason it overfits is three chime tunes, not twenty-four. ⚠️ The literal 24 class names are **unconfirmed** (arXiv PDF fetch returned binary) — grep the zip's folder names. |
| **Freesound API v2** | https://freesound.org/docs/api/ | Per-sound CC; **the API filters by license** — request CC0 + CC-BY only | >430 k clips | The actual answer to "where do I get 1000 doorbell clips". Every curated dataset above is a filtered snapshot of Freesound. Use the **content-based similarity endpoint** to expand from a seed set of good doorbells rather than trusting tag noise. Budget manual QC — "doorbell" returns plenty of phone rings. Example pack: https://freesound.org/people/kwahmah_02/packs/16967/ |

### 5.3 Filling `baby_cry` (we have ~151)

| Source | URL | License | Note |
|---|---|---|---|
| **gveres/donateacry-corpus** | https://github.com/gveres/donateacry-corpus | **unknown — verify** | Purpose-built infant cry corpus, phone-recorded in homes → closer to our deployment acoustics than FSD50K's curated clips. Filenames encode cry reason; collapse them all into `baby_cry`. Cry corpora frequently carry consent-linked restrictions. |
| **ICSD** | https://github.com/QingyuLiu0521/ICSD · mirror https://huggingface.co/datasets/QingyuLiu1/ICSD | **Research agreement: derivatives "must not be distributed or shared."** Internal research only | >3.3 h **strongly labeled** + ~1 h weak + synthetic. Strong onset/offset labels for infant cry are rare, and we classify 1 s windows — weak 10 s labels give us windows containing no cry at all labeled `baby_cry`. Ships source material so we can re-synthesize at 16 kHz with our own backgrounds. |
| **voc2vec** | https://huggingface.co/alkiskoudounas/voc2vec | **Apache-2.0** | Not data — a wav2vec2 foundation model for **non-verbal human vocalizations** (~125 h across 10 corpora incl. AudioSet vocalizations, FreeSound babies, Donate a Cry, TUT babies, NonSpeech7K, VocalSound, ReCANVo). Attacks the hard `baby_cry` failure mode: cry vs adult speech vs TV dialogue vs laughter. Its own eval list is a bibliography of cry corpora. |
| **CAPYLEE/CRSTC** | https://huggingface.co/CAPYLEE/CRSTC | **MIT** | Cry **detection** (cry vs not-cry) — the framing we need — not cry-reason classification. Ships BiLSTM, Transformer, **and MobileNetV2** checkpoints; the MobileNetV2 one is depthwise-separable + GAP + linear, near our op set. |
| **CryCeleb2023** | https://huggingface.co/datasets/Ubenwa/CryCeleb2023 | ⚠️ **CC-BY-NC-ND-4.0 and gated** | 26,000 files, 6.5 h of pure cry expirations, 786 infants, already 16 kHz. **NC + ND makes it unusable for a shipped product, arguably even for a distributed derived model.** Evaluation-only. Flagged so nobody spends a day on it. |

### 5.4 Filling `glass_break` (we have ~40 usable, from ESC-50)

| Source | URL | License | Note |
|---|---|---|---|
| **MIVIA Audio Events (Univ. of Salerno)** | https://mivia.unisa.it/datasets/audio-analysis/mivia-audio-events/ | ⚠️ **No license or terms published at all.** Access via request form https://mivia.unisa.it/datasets-request/ or mivia@unisa.it | By far the largest glass-break corpus available: 6000 events (4200 train / 1800 test), ~20 h train + ~9 h test, 32 kHz 16-bit. Each event at **6 SNRs (5/10/15/20/25/30 dB) over varied backgrounds** — exactly the axis a far-field battery INMP441 varies along. ⚠️ The 6000 counts SNR/background variants; **unique source recordings are fewer — ask them.** Do not plan a product around it until they answer in writing. |
| **TUT Rare Sound Events 2017** | §5.1 | NC | 139 isolated glassbreak events + synthesizer |
| **Edge Impulse #233502** **[V]** | https://studio.edgeimpulse.com/public/233502/latest | **BSD-3-Clause-Clear (data)** | 71 clips ≈ 3.5 min. Cheap top-up, not a fix. |
| **Zalmotek EI project #139844** | https://studio.edgeimpulse.com/public/139844/latest · writeup https://docs.edgeimpulse.com/experts/audio-projects/glass-break-detection-nordic-thingy53 | ⚠️ **unknown** — audio from unspecified "publicly available datasets" and license-free SFX sites | ~15 min glass-break + a **matched hard-negative background set (car horns, loud talking, doors closing)** — doors closing is exactly the confusion we care about. Reported 99.8% on first training run; treat with suspicion, a 2-class 15-minute set overfits trivially. |
| **GISE-51** | https://zenodo.org/records/4593514 · code https://github.com/SarthakYadav/GISE-51-pytorch | **CC BY 4.0 — shippable** | 16,357 **isolated** clips (12,465/1,716/2,176) over 51 FSD50K-vocabulary classes. Download only `isolated_events.tar.gz` (2.4 GB) + `noises.tar.gz` (64 MB) + `mixtures_jams.tar.gz` (86.5 MB) → a Scaper-compatible mixing pipeline under CC BY. Ships pretrained baselines (2.6 GB). ⚠️ The literal 51 class names are **unconfirmed** — read `meta/lbl_map.csv`. |

⚠️ **Structural warning on `glass_break`.** Glass-break acoustic energy spans from below 3 Hz to well over 20 kHz, and commercial detectors reduce false alarms by requiring **multiple frequency components**. Our front end is band-limited to 20–7800 Hz at 16 kHz sampling, so **we are structurally blind to the high-frequency shard content commercial detectors rely on.** That is a plausible concrete contributor to 0.15 recall, and it is a **sample-rate decision, not a model decision**. The SIA standard that presumably defines a test signal — https://www.securityindustry.org/industry-standards/gb-01-2014/ — returned 403 and could not be read [S].

### 5.5 Filling `fire_alarm` (there is essentially no open corpus — synthesize)

There is **no substantial dedicated open fire-alarm *audio* dataset**. Nearly every GitHub result for "fire alarm detection" is visual flame/smoke detection; the Kaggle "smoke detection" sets are IoT gas/temperature tabular data. **Synthesis from the T3/T4 standard is the right answer** (shortlist #10). Supplement with:

- **NIGENS** — https://zenodo.org/records/2535878 — ⚠️ **CC BY-NC-ND 4.0** (non-commercial **and** no-derivatives). 1017 WAVs (714 isolated events across 14 classes + **303 'general' catch-all**), ~4 h 46 m, per-file `.txt` event timings, 2.2 GB. Has an explicit `alarm` class, `crying baby`, and `crash` (partial glass). The 303 'general' files are the closest public analogue to our `background` definition. **ND clause: teacher/eval side only, never in shipped training data.**
- **ProtoSound** — https://github.com/makeabilitylab/ProtoSound — **MIT**. 10 categories hand-curated by researchers *specifically for deaf/HoH home alerting*, explicitly including fire alarm, knocking, baby crying. Small, but curation matters more than volume — mislabeled fire-alarm clips are plausibly part of why we sit at 0.13. Audio in `StandaloneAndroidApp/app/src/main/assets/library`.
- **DESED `alarm_bell_ringing`** — https://github.com/turpaultn/DESED — per-file Freesound licenses (mixed CC0/CC-BY/CC-BY-NC, must be filtered). 1009 foreground train files + 314 eval. ⚠️ DESED **lumps doorbells, alarm clocks and alarm bells into one class** — manual re-split required into our `doorbell` vs `fire_alarm`.

### 5.6 What we already have, re-examined

- **FSD50K** — https://zenodo.org/records/4060432. **The per-clip license split matters and we may already be non-compliant**: dev = 14,959 CC0 / 20,017 CC-BY / **4,616 CC-BY-NC** / 1,374 CC Sampling+; eval = 4,914 CC0 / 3,489 CC-BY / **1,425 CC-BY-NC** / 403 Sampling+. That is **6,041 NC clips across dev+eval**, plus an explicit "contact Eduardo Fonseca and Frederic Font for commercial purposes" instruction on the record. Per-clip mapping is in `dev_clips_info_FSD50K.json` / `eval_clips_info_FSD50K.json`. **If we trained on all of it, the current model is not shippable.**
- **ESC-50** — https://github.com/karolpiczak/ESC-50. **CC BY-NC 3.0** (the **ESC-10 subset is CC BY**, attribution only). 40 clips each of `crying_baby` and `glass_breaking` — which is exactly what we have and exactly why those recalls are 0.15. Remaining value: domestic negatives (washing machine, vacuum, keyboard, clock tick, mouse click) for `background`, and as the standard benchmark to report against.

---

## 6. The `tv_music` = 0.02 / `background` = 0.001 problem

**This is our worst number and it is also the one the literature is clearest about. Read this section first if you read only one.**

### 6.1 First: what these numbers actually are

**Read the §1 correction first.** 0.02 and 0.001 are at-gate figures; at argmax `tv_music` is **0.397** and `background` is **0.571**, and `background` is the most-predicted class of the six. So this section is about a real weakness, not a phantom one — but the weakness is ordinary under-training on the negative classes, and the prescriptions below are sized accordingly.

Three lines of evidence on how much room there is:

1. **Against benchmarks.** On DCASE 2023 Task 4A domestic SED — 10 classes, GPU-scale models, ensembling — the **best PSDS1 is 0.625, best single system 0.591, and the published baseline is 0.327** (https://dcase.community/challenge2023/task-sound-event-detection-with-weak-labels-and-synthetic-soundscapes-results [F]). The task is genuinely hard, ~0.6 is state of the art, and **we should not target 0.95**. Our 0.397 argmax on `tv_music` is a recognisable early number for this problem, not an anomaly.
2. **Against the negative-class taxonomy in the reference implementations.** mlcommons/tiny's `streaming_wakeword` uses `--num_classes {2,3}` over `target / unknown / silent`, and its reference eval set is **2,796 targets / 1,398 silent / 9,786 other — negatives outnumber positives roughly 4:1** [V]. Arm KWS uses 10 words + `_silence_` + `_unknown_`. **Check whether our training mix looks anything like that.** If it does not, that alone is a candidate explanation.
3. **Against the one measured price tag we have.** ST's own FSD50K yamnet drops from **87.0% (without unknown) to 73.9% (with unknown) on the same dataset** just by adding an open-set class [V]. Adding a `background` class is expensive by construction; that is a real 13-point tax, not a bug. But it takes you to 74%, not to 0.001.

### 6.2 What the low-complexity ASC and wake-word literature actually says to do

**(a) Negative mass, at a ratio nobody expects.** `openWakeWord` — https://github.com/dscripka/openWakeWord [F] — is the best open illustration. Code **Apache-2.0**; ⚠️ **pretrained models are CC BY-NC-SA 4.0** because of training-data licensing, so lift the pipeline, not the models. The people who solved false-wake-on-TV did it with roughly **30,000 hours of negative data** representing speech, noise and music — a ~200:1 negative-to-positive ratio — **not a cleverer architecture**. Our `background` at 0.001 and `tv_music` at 0.02 are the signature of a class with almost no training mass. openWakeWord also uses the **Dinner Party Corpus** as a false-accept benchmark (https://zenodo.org/records/8122551, CDLA-Permissive per search results [S] — 10 sessions × 15–45 min, one close-talk mic plus **five far-field 7-mic arrays at different room positions**, i.e. our exact geometry).

**(b) Real TV, in a real room, not clean music.** This is the specific gap. Our `tv_music` positives are presumably clean studio music and clean speech; the deployment sees a loudspeaker across a room. Sources, in order of how well they match:

| Source | URL | License | Why |
|---|---|---|---|
| **DCASE 2018 Task 5 / SINS** | https://zenodo.org/records/1247102 | **CC BY-NC 4.0** | Explicit hand-annotated **`Watching TV`** class in a real living-room/kitchen with 4 fixed mic arrays, ~200 h, plus **`Absence`** (empty room) — genuinely loudspeaker-reproduced-in-a-room audio, which is the discriminative cue. Also `Cooking, Dishwashing, Eating, Social activity, Vacuum cleaning, Working` as hard negatives. Re-cut from the continuous upstream (https://github.com/KULeuvenADVISE/SINS_database) rather than the 10 s segments to control window alignment ourselves. |
| **CHiME-Home** | https://archive.org/details/chime-home | **CC BY-NC-SA 3.0** | 6.8 h domestic, 6,137 × 4 s chunks, **1,946 with strong annotator agreement**, and one of its 7 tags is literally **`video game / TV`**. Small enough (3.9 GB) to download today. 4 s chunks → 4 non-overlapping 1 s windows each, or ~16 at our 4 Hz hop. |
| **MUSAN** | https://www.openslr.org/17/ | **CC BY 4.0 — shippable** | 109 h, already 16 kHz mono. `music` (~660 files, multi-genre) + `speech` (~426 files, 12 languages, LibriVox + US Govt hearings) = `tv_music`; `noise` (~930) = `background`. **Convolve with room IRs.** |
| **FMA** | https://github.com/mdeff/fma | Metadata CC BY 4.0; **audio per-track CC, must be filtered** | 106,574 tracks / 161 genres. Use `fma_small`, not the 917 GiB full set. Genre breadth is what stops `tv_music` overfitting to one style. **Again: convolve with RIRs — training on clean studio music is probably part of why `tv_music` is 0.02.** |
| **TVSM (Netflix)** | https://github.com/biboamy/TVSM-dataset/ · https://zenodo.org/records/7025971 | **Apache-2.0** | 1,608 h of professionally produced TV audio with **frame-level speech/music labels**. ⚠️ **No raw audio is released** — only precomputed mel/MFCC/VGGish features + CSV labels, and their mel config is not our 40-band / 49-frame / 20–7800 Hz HTK setup, so **the features are unusable as training input**. What transfers: two pretrained CRNN checkpoints (`CRNN-P-Cue`, `CRNN-P-Pseu`) as a pseudo-labeler for TV audio we record ourselves, and the labeling methodology. |
| **OpenBMAT** | https://zenodo.org/records/3381249 | ⚠️ **Request-gated, non-profit purposes only** | 27.4 h of real TV broadcast, 1,647 one-minute excerpts, 8 program types, 4 countries — and uniquely with **relative-loudness annotations** (music foreground / music background / no music), cross-annotated by 3 people. Maps onto our hardest real case: **TV playing quietly under a conversation.** Lets us build a curriculum by music loudness. Eval / pseudo-label target only. |

**(c) Channel and device mismatch.** We train on FSD50K/ESC-50 (varied mics, often close-mic'd, the sound isolated as the subject of the recording) and deploy on **one specific INMP441 in a room**. `theMoro/DIRAugmentation` — https://github.com/theMoro/DIRAugmentation [F], ⚠️ **no LICENSE stated** — convolves training audio with device impulse responses; the paper (arXiv 2305.07499) reports it matches Freq-MixStyle in isolation and that the two are **complementary**, best combined on unseen devices. IRs come from MicIRP (http://micirp.blogspot.com/), CC with attribution — but it is a **Blogspot site with per-microphone posts, not a packaged dataset**, and those are **vintage studio microphone** IRs, not MEMS. **For an INMP441 we will get more mileage measuring our own device IR.** Implementation layer: `audiomentations` / `torch-audiomentations` (https://github.com/iver56/audiomentations, **MIT**, GPU-batched `nn.Module` transforms).

**(d) Pretrain on a larger in-domain scene corpus, then distill.** The DCASE 2025 Task 1 rank-1 system (Karasin_JKU, 61,148 params / 29,419,156 MACs, 61.5%) used **domain-specific pretraining on CochlScene followed by multi-teacher distillation from CP-ResNet + BEATs + PaSST** (https://dcase.community/challenge2025/task-low-complexity-acoustic-scene-classification-with-device-information-results [F]). CochlScene — https://zenodo.org/record/7080122, ⚠️ **CC BY-SA 3.0 (share-alike — check compatibility)** — is 76,115 × 10 s files from 831 crowdsourcing participants on **consumer phones**, a far better channel match for an INMP441 than TAU's professional rig. ⚠️ The 13 class names are **unconfirmed**; read `Data_info.tsv`. Every top DCASE system across 2024 and 2025 used **knowledge distillation + quantization + pruning; none trained a tiny model from scratch, which is what we are doing today.**

**(e) Receptive-field regularization** — `kkoutini/cpjku_dcase20`, https://github.com/kkoutini/cpjku_dcase20 [F], ⚠️ **no license stated**, weights at https://zenodo.org/record/4282667. Its **decomposed CP-ResNet is 18,740 trainable params (34.2 kB in fp16) and hit 95.83% dev accuracy on a 3-class scene task** — *smaller than our current model*. The idea is deliberately capping the receptive field so the net cannot memorize a whole clip. Directly relevant: `background` and `tv_music` are defined by **long-horizon texture**, and an unconstrained tiny CNN on a 49-frame window will latch onto transients instead. Read `models/cp_resnet_decomp.py` and `cp_resnet_prune.py`.

**(f) The domain gap nobody wants to hear about.** Cochl's own docs carry an explicit warning that their tags are **optimized for real sound and not sound played back by laptops, smartphones, or speakers** (https://docs.cochl.ai/sense/cochl.sense-cloud-api/soundeventdetection/soundtags/ [F]). A commercial vendor is telling us there is a measurable gap between live acoustic events and replayed recordings. **Our training data is FSD50K + ESC-50 — overwhelmingly replayed/recorded material — and our deployment is an INMP441 hearing live events in a room.** Corroborating anecdote: press testing reported an Echo failed to detect real glass smashed ~3 feet away while responding to glass-break sound *effects* played from a phone [S]. This plausibly contributes more to `fire_alarm` and `glass_break` recall than architecture does.

### 6.3 Measure the right thing

Our current per-class recall table **cannot show the number that decides whether this ships.** Adopt `psds_eval` (**MIT**, https://github.com/DCASE-REPO/psds_eval, paper https://arxiv.org/abs/1910.08440). It gives effective **false positives per hour** and, critically, separates **cross-triggers** (our `tv_music` detector firing on a real doorbell) from generic false positives, across operating points rather than at one threshold. With two explicit negative classes whose entire job is to suppress the positives, **cross-trigger rate is the number that tells us whether `tv_music` and `background` are doing their job.**

The published shipping bar, for calibration: **Sensory states they work a sound until it exceeds 90% detection at 1 false alarm per 24 hours** (https://sensory.com/identifying-sounds-as-accurately-as-wake-words/ [F]). Their models are "around 100 kBytes", combined two-stage under 5 MB, average power under 2 mA. **Two takeaways:** our 35 kB is the right order of magnitude, not off by 10×; and their **two-stage architecture** — a cheap always-on gate plus a heavier confirmer that "eliminates 95% of false alarms from the first stage while passing 97% of real events" — fits our headroom better than scaling one model.

### 6.4 The candid vendor evidence

- **Axis** (https://www.axis.com/dam/public/07/72/4f/audio-analytics-for-security-and-safety-en-US-431871.pdf, extracted to `…/scratchpad/axis_audio_analytics.txt` [F]) states plainly that their analytics may produce false alerts with a lot of background noise, and names the culprits: rain on windows, thunder, sirens, **music, and busy scenes with people talking**. Their recommended deployment is therefore **quiet areas** — banks, reception desks, indoor spaces after hours. Read that as: a vendor with a camera-class compute budget could not solve the music-and-talking negative and **scoped the product to quiet rooms instead.** Our instinct that a system alerting on the television is worse than no system is correct, and it is the genuinely hard part of this project — not the six-way classification. Axis also shipped scream/shout **first** and deferred glass break.
- **Google Sound Notifications** ships exactly nine categories — smoke/fire alarms, sirens, baby sounds, dog barking, knocking, doorbell, appliance beeping, landline phone, water running (https://support.google.com/accessibility/android/answer/10092548 [F]) — and **deliberately omits glass breaking**, with no TV/music negative at all.
- **Apple Sound Recognition** ships fire alarms, sirens, smoke alarms, doorbells, door knocks, **glass breaking**, baby crying, and more (https://support.apple.com/guide/iphone/use-sound-recognition-iphf2dc33312/ios [S]) — **four of our five positives** — and offers **user-trained custom sounds specifically for alarms, doorbells and appliances**. That last detail is a product hint: Apple is admitting doorbells vary too much between homes to cover with one general model. With ~144 doorbell clips, **an on-device few-shot enrollment path may be a better product answer than a universal doorbell detector.**
- **Aizip / Seeed SED Module D1** (https://aizip.ai/models/aed, https://wiki.seeedstudio.com/sound_event_detection_module/ [F]) ships exactly `Baby Cry / Glass Break / Gunshot / Smoke Alarm & CO Alarm (T3/T4) / Snoring` in **100–300 kB ROM, ~50 kB RAM, >95% detection, <2% FP, ≤500 ms latency** on an XMOS XU316. Nothing to lift — but it brackets our budget almost exactly and is the hardest existence proof that **our spec is achievable at our footprint, so our metrics are a data/recipe problem**. Note their class name literally says **T3/T4** — a shipping commercial product naming the ANSI temporal patterns in its class definition, independently confirming that `fire_alarm` is detected by **cadence, not timbre**.
- **Syntiant NDP101 ships inside the Amazon Ring Alarm Glass Break Sensor** (https://www.edge-ai-vision.com/2023/01/syntiant-to-introduce-turnkey-edge-ai-security-solution-at-ces-2023/ [F]), stated 25 ft detection range. Amazon put **dedicated neural silicon in a single-purpose glass-break sensor**. That does not make our ESP32-S3 approach wrong, but `glass_break` is where the industry felt it needed dedicated hardware — set expectations, and consider whether that class specifically deserves Sensory's two-stage treatment. **Range in feet is probably a more honest metric for us than clip-level recall.**

---

## 7. Licensing — what we can and cannot ship

"Ship" = the artifact or data materially contributes to weights we distribute in firmware. Using a model **offline as a teacher** and shipping only a student is a **legally unsettled** question in every case; the common industry position is that trained weights are not a derivative of the training corpus, but that position is not settled. Where a corpus carries an NC clause **and** an explicit "contact us for commercial use" instruction (FSD50K), get counsel.

### Ship-safe

| Artifact | License | Obligation |
|---|---|---|
| EfficientAT weights (`mn01…mn40`, `dymn`) + PaSST ensemble logits | **MIT** | Retain notice |
| PretrainedSED | **MIT** | Retain notice |
| CED weights on HF (`mispeech/ced-*`) | **Apache-2.0** | Retain notice. ⚠️ **Not** the `RicherMans/CED` repo (GPL-3.0) |
| ConvNeXt-Tiny AudioSet (`topel`) | **MIT** | Retain notice |
| ST `miniresnetv2` / `yamnet` / `miniresnetv1` `ST_pretrainedmodel_public_dataset` **[V]** | **Apache-2.0**, verbatim, **no ST-device clause** | Retain notice. ⚠️ Repo-level metadata is `NOASSERTION` — grant is scoped to those subfolders. ⚠️ FSD50K corpus question below. |
| Arm ML-zoo KWS, ML-KWS-for-MCU, mlcommons/tiny **[V]** | **Apache-2.0** | Retain notice + NOTICE. Speech Commands (CC-BY 4.0), **not** AudioSet — no NC trap |
| AST (`MIT/ast-finetuned-audioset-*`) | **BSD-3-Clause** | Retain notice |
| Audio-MAE | **CC-BY 4.0** | Attribution |
| PANNs weights (Zenodo 3987831) | **CC BY 4.0** (code MIT) | **Attribution attaches to shipped derivatives** |
| YAMNet weights | **CC BY 4.0** per MediaPipe (repo code Apache-2.0) | Attribution |
| MUSAN | **CC BY 4.0** | Attribution |
| GISE-51 | **CC BY 4.0** | Attribution |
| Zenodo 3689288 (24 alarm classes) | **CC BY 4.0** | Attribution |
| VGGSound | **CC BY 4.0**, VGG page states commercial+research OK | Attribution. ⚠️ **Class list unverified** — see §10 |
| FSDnoisy18k | **CC-BY or CC0 per clip** | Per-clip attribution |
| agkphysics/AudioSet | **CC-BY 4.0** | Attribution |
| AudioSet weak + strong label TSVs; Zenodo 7096702 reformat | **CC-BY 4.0** | Attribution |
| Edge Impulse #237590 doorbell **data** | **Apache-2.0** | Retain notice |
| Edge Impulse #233502 glass **data** **[V]** | **BSD-3-Clause-Clear** (no patent grant) | ⚠️ No README/attribution/provenance — EI can only sublicense what it owns |
| LAION-CLAP (repo CC0 / ckpt Apache-2.0), Microsoft CLAP repo MIT | permissive | ⚠️ HF `microsoft/msclap` weights showed `ms-pl` — verify |
| `psds_eval`, `DESED_task`, `audiomentations`, `inaSpeechSegmenter`, `silero-vad`, `ProtoSound`, `CAPYLEE/CRSTC`, `voc2vec` | MIT / Apache-2.0 | Retain notice |
| `google-research/sound-separation` (code **and** checkpoints) | **Apache-2.0** | Retain notice |
| `yqcai888/easy_dcase_task1` | **Apache-2.0** | Retain notice |
| TVSM | **Apache-2.0** | ⚠️ Features only, no raw audio — see §6.2 |
| AudioSet ontology JSON | **CC BY-SA 4.0** | ⚠️ ShareAlike governs the **file**; adopting the label *strings* as internal class names is a much weaker use than redistributing it |
| ANSI T3/T4 **pattern** | unencumbered knowledge | The standard *document* is paid; the cadence is not |

### Cannot ship

| Artifact | License | Why |
|---|---|---|
| **FSD50K** | Mixed per-clip: **6,041 CC-BY-NC across dev+eval**, plus Sampling+, plus an explicit "contact for commercial purposes" | **Must filter** via `dev_clips_info_FSD50K.json` / `eval_clips_info_FSD50K.json`. **If we trained on all of it, the current model is not shippable.** Also the residual question hanging over ST's FSD50K-pretrained backbone. |
| ESC-50 | **CC BY-NC 3.0** (ESC-10 subset is CC BY) | NC |
| TUT Rare Sound Events 2017 (Zenodo 401395) | **Other (Non-Commercial)** | NC |
| TAU Urban Acoustic Scenes 2020 / 2022, TUT Acoustic Scenes 2016 | **Other (Non-Commercial)** | NC — architecture validation only |
| DCASE 2018 Task 5 / SINS | **CC BY-NC 4.0** | NC |
| CHiME-Home | **CC BY-NC-SA 3.0** | NC + SA |
| NIGENS | **CC BY-NC-ND 4.0** | NC **and ND** — arguably blocks distributing a derived model |
| CryCeleb2023 | **CC-BY-NC-ND-4.0**, gated | NC + ND |
| OpenBMAT | Request-gated, **non-profit only** | |
| ICSD | Research agreement: derivatives **must not be distributed** | |
| openWakeWord **models** | **CC BY-NC-SA 4.0** | Code is Apache-2.0 — lift the pipeline, not the models |
| **Edge Impulse model exports (free tier)** **[V]** | Developer plan ✗ "External distribution to third parties"; production ≤1000 units requires Enterprise Production Phase | Verified at https://edgeimpulse.com/pricing. **Never ship an EI export on the free tier.** Data license ≠ export license. |
| `RicherMans/CED` code, `gbibbo/ai4s-embedded` | **GPL-3.0** | Viral. Do not link into firmware. Weights are separately licensed for CED. |
| DEMAND (inside MS-SNSD) | **CC-BY-SA 3.0** component | Share-alike |
| CochlScene | **CC BY-SA 3.0** | Share-alike — check compatibility before shipping weights |
| AudioSet **audio** | YouTube content, not redistributed | Scraping has ToS implications for a commercial product. Use `agkphysics/AudioSet` (CC-BY-4.0) instead of `yt-dlp`. |

### Unknown — do not assume

`gveres/donateacry-corpus` · MIVIA (no terms published at all) · `theMoro/DIRAugmentation` · `Qualcomm-AI-research/bcresnet` · HTS-AT · OpenL3 · `kkoutini/cpjku_dcase20` · CPJKU dcase2024/2025 Task 1 baselines (LICENSE URL is a hard 404) · `fschmid56/cpjku_dcase23` (**no LICENSE file at all → all rights reserved by default**) · `gentonesv4b.zip` · `k2-fsa` sherpa-onnx model artifacts · `litert-community/PANNs-CNN14-AudioSet-LiteRT` · Zalmotek EI #139844 audio · DESED aggregate.

---

## 8. What we'd do first

Ordered. Someone can start on step 0 tomorrow at 9am with no downloads.

### Step 0 — ~~Falsify the bug hypothesis~~ Already done; here is what it found (half a day, mostly spent)

**This step was run.** The hypothesis was that `background` at 0.001 meant a mis-wired label map. It does not — see the correction in §1. The reported figures are at the 75% gate; at argmax `background` recall is **0.571** and it is the most-predicted class of the six. Argmax accuracy 0.494, shipped `temperature` 0.924. No wiring bug exists. **Do not spend the morning looking for one.**

What that leaves, which is the real finding: **the model is under-confident because it is weak**, so the gate — not the classifier — is what produces the 0.001. Two consequences for everything below:

- The 75% gate is doing enormous work and nobody has measured its cost. Sweep it. Plot recall and false-alarms-per-hour against the threshold *per class* rather than assuming one gate suits both a fire alarm and a television. `fire_alarm` at 0.464 argmax and 0.125 at-gate means roughly **three-quarters of the fire alarms the model already finds are being thrown away by the threshold.**
- Because the deficit is capacity and data rather than plumbing, §4/§5/§8-Step-4 are the fix, and they are worth starting immediately.

Still worth doing from the original step, both cheap:

1. Check the negative:positive ratio. mlcommons/tiny's `streaming_wakeword` reference eval is **~4:1 negative-heavy** [V]; Arm KWS carries `_silence_` + `_unknown_` as first-class labels. If ours is not remotely negative-heavy, that is the finding.
3. Stand up a **fourth front-end parity oracle**. Clone `mlcommons/tiny`, take `benchmark/training/keyword_spotting/get_dataset.py:105-125`, make three edits — `upper_edge_hertz = 7800.0`, `--window_size_ms 40`, delete the `mfccs_from_log_mel_spectrograms(...)[..., :dct_coefficient_count]` line — and diff its `(49, 40)` output (one transpose from our `(40, 49)`) against our PyTorch front end on the same WAVs. Apache-2.0, benchmark-audited, ~1 hour. Decide explicitly whether to also switch `tf.abs(stfts)` to a power spectrum to match us.
4. Read `mlcommons/tiny`'s `quantize.py` + `quant_cal_idxs.txt` — a worked example of choosing int8 calibration samples from a **curated index list**. Our `fire_alarm` 0.13 / `glass_break` 0.15 are consistent with per-output-channel scales calibrated on a sample that under-represents those classes. Cheap to check.

**If step 0 finds the bug, most of steps 1–4 gets cheaper, not unnecessary.**

### Step 1 — Get the teacher and build the six-class mapping (1 day)

```
curl -O https://storage.googleapis.com/audioset/yamnet.h5     # 521-class map + baseline
pip install transformers && # mispeech/ced-base, Apache-2.0 weights, variable-length input
```

- Take the **AudioSet ontology** (https://github.com/audioset/ontology, 632 nodes, 342,780 B JSON) and write the six-class mapping as an **explicit many-to-one table over `/m/` machine IDs**, not over class indices. Every AudioSet teacher emits these IDs; a mapping written over IDs survives swapping CED for PANNs for EfficientAT, and mines FSD50K with the same table. Verified IDs: `doorbell ← /m/03wwcy, /m/07r67yg` · `fire_alarm ← /m/01y3hg, /m/0c3f7m` (hold out `/m/07pp_mv Alarm`, `/m/030rvx Buzzer`, `/m/046dlr Alarm clock`, `/m/02mfyn Car alarm` as **negatives**) · `baby_cry ← /t/dd00002, /m/0463cq4` · `glass_break ← /m/07rn7sz, /m/039jq, /m/07pc8lb` · `tv_music ← /m/07c52, /m/04rlf, /m/025td0t, /m/06bz3, /m/09x0r` · `background ← /m/028v0c Silence` + everything else.
- ⚠️ **YAMNet is 521-class ordering; PANNs/CED/EfficientAT are 527-class. The indices differ.** Mapping over `/m/` IDs is the whole point.
- **Primary teacher: CED** (Apache-2.0 weights, accepts our real 1 s windows). **Cross-check: AST** (BSD-3). Keep only pseudo-labels where both agree.
- In parallel, run **Apple SoundAnalysis** over the same audio (`…/scratchpad/classify.swift`, ~214× real time, free, offline) as a third opinion — 5 of 6 classes map directly; compose `tv_music` from `music`+`speech`+`chatter`.

### Step 2 — Fix the window alignment, then fix `fire_alarm` for almost nothing (2–3 days)

- **`fire_alarm` first because it is nearly free.** Synthesize T3/T4 from spec (3 × 0.5 s pulses / 0.5 s gaps / 1.5 s pause, ~3 kHz piezo), sweeping carrier frequency, harmonics, sweep-vs-steady, RIR, distance/SNR, **and phase offset relative to the 1 s window**. At 4 Hz some windows contain three beeps and some contain the 1.5 s silence; if training never saw the silence-heavy windows, that alone explains 0.13. Reference generator (reimplement rather than vendor — no license): https://goughlui.com/2024/01/29/project-generate-high-quality-industrial-fire-alarm-sounder-audio-with-python/. Fixtures already at `…/scratchpad/t3_smoke.wav`, `t4_co.wav` — use them as unit-test fixtures for all three runtimes.
- Then add the **cadence post-filter**: a ~5 s ring buffer of the per-frame `fire_alarm` score (20 floats/class at 4 Hz, <1 kB RAM) with autocorrelation or template match against the T3/T4 period. Pure C, no retraining, no new ops. A shipping commercial product (Aizip/Seeed) names T3/T4 in its class definition; this is how the class is actually detected.
- **Then `glass_break` and the transients.** Run **PretrainedSED** (MIT, `frame_mn06`/`frame_mn10`) over our existing corpus and any bulk audio to get ~0.1 s onsets, and re-cut every alert-class window centred on the actual transient. Cross-check against **AudioSet-Strong** human labels (https://zenodo.org/records/7096702, 934,821 events over 103,463 clips, CC-BY-4.0 TSVs, no audio) paired with `agkphysics/AudioSet` for the audio.

### Step 3 — Fix the negative classes, which is where our worst numbers are (3–5 days)

1. `wget` **MUSAN** (11 GB, CC BY 4.0, already 16 kHz mono). `music` + `speech` → `tv_music`; `noise` → `background`.
2. **Convolve everything with room impulse responses before training** and mix at varied SNR. Use `torch-audiomentations` (MIT, GPU-batched). Measure our own INMP441 device IR rather than using MicIRP's vintage studio-mic IRs.
3. Pull **DCASE 2018 Task 5 / SINS** (`Watching TV` + `Absence`) and **CHiME-Home** (`video game / TV`, 1,946 strong-agreement chunks, only 3.9 GB) as **pseudo-label / eval material only** — both NC, neither can back shipped weights.
4. **Mine hard negatives, which is the step that actually moves 0.02 and 0.001:** windows where the teacher says `background`/`tv_music` but our current student fires. Target an openWakeWord-scale negative:positive ratio, not a balanced one.
5. Use `google-research/sound-separation` (Apache-2.0) to split mined clips and **re-mix at controlled SNR into hard `doorbell`-under-`tv_music` pairs** — the exact confusion the product cannot afford.

### Step 4 — Change how we train and how we measure (1 week)

- **Distill, don't train from scratch.** Every top DCASE system across 2024 and 2025 used KD + quantization + pruning; none trained a tiny model from scratch, which is what we do today. Loss = `CE(hard) + α·KL(student 6 logits ‖ teacher's 6 mapped group logits, T≈3)`. **The teacher consumes 16 kHz waveform and computes its own front end internally, so the geometry mismatch never touches device code** — keep our `(40, 49)`, HTK, 20–7800 Hz, 1e-6 floor, and our conv/ReLU/GAP/linear layer table. This dodges the depthwise port, the fp32 RAM blowup, and the front-end rewrite all at once.
- **Adopt `psds_eval`** (MIT) and start reporting **false positives per hour** and **cross-trigger rate**, not per-class recall. Target Sensory's published bar: **>90% detection at ≤1 false alarm per 24 hours.**
- **Only then** consider the architecture. If we do: initialize from ST's `pooled_miniresnetv2_1_stacks_backbone.keras` (Apache-2.0, GAP-terminated), **cut conv1 from 64 to ~32 filters before training** to fit fp32 activations, add element-wise Add + MaxPool2D to the layer table **as a DAG with a second live buffer**, and mirror all of it in C, JS and PyTorch for parity. Consider Sensory's **two-stage** shape — a cheap always-on gate plus a heavier confirmer — before scaling one model.

---

## 9. Dead ends — do not re-walk these

**Wrong thing entirely**
- `natyavidhan/esp-infer` — snippets say "glass break, smoke alarms, baby cries"; it is an **MPU6050 gesture classifier** (668 params, dense-only MLP). Zero audio.
- Edge Impulse #31378 "Smoke Detector" — **CO2/TVOC/humidity gas sensor**, 28 samples, labels ambient/meat/smoke. Not audio.
- Kaggle "Smoke Detection Dataset" and most GitHub "fire alarm detection" — IoT tabular sensor data or **computer-vision flame/smoke**. There is no substantial open fire-alarm *audio* dataset; synthesize from T3/T4.
- Arm ML-zoo `anomaly_detection` MicroNet — DCASE machine-condition monitoring (industrial fans/pumps/valves). No overlap.

**No acoustic-event models in the ESP ecosystem**
- `espressif/esp-dl` model zoo is **vision-only** (YOLO11n, pose, seg, face, cat, dog). No acoustic event model.
- `espressif/esp-sr` is WakeNet / MultiNet / VADNet / AFE / TTS — **speech only**, chip-specific binaries, not portable weights.
- Every "ESP32 + INMP441" repo is an FFT spectrum analyzer, SPL meter, or I2S recorder (`ESP32-INMP441-SPECTRUM`, `ESP32-Audio-Analyzer`, `esphome_sound_source_detection`). The one ML repo, `happychriss/edgeML_esp32_audio_sampling`, is spoken digits 0–9.
- `nyumaya/nyumaya_audio_recognition` — inference lib is Apache-2.0 and lists ESP32S3, but the only free models are **hotwords** (including a "View Glass" *spoken phrase*, not glass break). Sound-event models are commercial-on-request.
- Syntiant NDP120 / Arduino Nicla Voice — **no downloadable pretrained sound-event weights exist**; models must be trained through Edge Impulse's Syntiant blocks and compiled for NDP120 silicon. Multiple forum threads are people asking for exactly this and not getting it.
- Seeed SED Module D1 — proprietary Aizip binary on XMOS. Host-side Arduino library only. Buyable hardware, not liftable weights.
- `tflite-micro` `micro_speech` (<20 kB) — right footprint tier, but Speech Commands, and we have ruled out TFLite Micro.

**Refuted after fetching**
- `fschmid56/cpjku_dcase23` — pitched as "the single most liftable recipe." The checkpoints are real, but you **cannot** swap the 10-class urban head for ours: the auto-downloaded logits are a **10-dim tensor positionally indexed into the TAU22 train split** (`AddLogitsDataset` does `self.logits[self.map_indices[index]]`) and all 12 teachers end in a 10-class urban head. Change the label set and every weight and logit is worthless. **No LICENSE file at all** (all rights reserved), and every checkpoint derives from TAU22, marked "Other (Non-Commercial)". CP-Mobile also needs grouped/depthwise conv, residual add, strided AvgPool2d and **GRN** (whose L2 spatial norm the repo itself has to dequantize around). The useful discovery from checking it was the **same author's `EfficientAT`**, MIT and AudioSet-pretrained.
- Edge Impulse #149095 "Vandalism Detection" (CodersCafe) — real, **Apache-2.0**, int8 419 ms / 40.7K / 32.2K published. But **~2,970 params — ~10× smaller than what we already run** — with MaxPooling1D + Flatten→Dense (ops we lack) on a 40×199 / 20-10 ms / 300 Hz-floor front end. Its GitHub repo is 3 files / 33 KB with no LICENSE and no model. Its negative set is **MS-SNSD**, which I enumerated: **no music and no television whatsoever**, so it does nothing for `tv_music` 0.02. Test set is 34 clips. Training clips were re-recorded through a speaker into an MP34DT05 PDM mic, not an INMP441. **Clone MS-SNSD from Microsoft directly for its CC0 subset; never open the EI project.**

**Fetch/tooling failures — save the round-trip**
- Kaggle Models YAMNet page renders as a JS shell; returns only breadcrumbs. Use `tensorflow/models` + `storage.googleapis.com/audioset/yamnet.h5`, or the MediaPipe `.tflite` link.
- `k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models` — assets list fails server-side. Use the HF repos + tree API for exact byte sizes.
- arXiv 2211.04772 (EfficientAT) — no results table on the abstract page, PDF is unparseable binary, no HTML rendering for any version. **The GitHub Releases API is strictly better: the filenames encode the mAP.**
- arXiv 2002.11561 (Zenodo 3689288's paper) and the arXiv PDFs generally return raw binary through WebFetch. The 24 alarm-class names are still unconfirmed — read the zip's folder names.
- `raw.githubusercontent.com/hche11/VGGSound/master/data/class_labels_indices_vgg.csv` → **404**. Get `vggsound.csv` from the VGG page and grep it.
- `wiki.thefirepanel.com` → 403. `securityindustry.org` → 403. Neither exact T3/T4 timing nor SIA GB-01-2014 could be read from primary sources.
- Cochl's tag tables render client-side — both WebFetch and curl return only category headers. Their Edge SDK "Benchmark" page is 404.
- The Axis white paper PDF must be `pdftotext`'d locally; WebFetch cannot read it.
- `github.com/audioanalytic/psds_eval` → **404**. Use `DCASE-REPO/psds_eval`.
- `audioanalytic.com` — **DNS does not resolve.** Acquired by Meta Nov 2022; the whole site went with it, including the `ai3-nano` "sound recognition at 40 kB" page (which would have been the closest published analogue to our 35 kB budget). `web.archive.org` was blocked in this environment. Their **"Alexandria"** dataset (40 M recordings, 1,200 classes) was **never publicly released** and is now inside Meta — do not look for it.

**No such artifact exists**
- **No pretrained, downloadable audio classifier with a ready-made household head matching our six classes.** Hugging Face `audio-classification` searched specifically for doorbell/fire-alarm/smoke-alarm household models: only general AudioSet taggers and speech-command models. **We build the head ourselves; the AudioSet ontology subset is the bridge.**
- **No AudioSet tagger already fine-tuned down to a small home-safety class set.** Everything is either the full 527/521-class head or a generic embedding. The class-collapse step is ours.
- **No open-source "TV audio detector" with downloadable weights.** The commercial work (Amazon/Sonos wake-word suppression) is published as **patents** (US 11900937, 12340802, 11380322, 10475449) and papers with no released models or data. Closest substitutes: SINS `Watching TV`, CHiME-Home `video game/TV`, `inaSpeechSegmenter`, AudioSet `Television` (index 518).
- **No open live-vs-loudspeaker corpus for general non-speech audio**, and no pretrained replay-detection weights with a usable license. The ASVspoof PA angle is less transferable than it sounds: high-quality loudspeakers on **both** sides in places, speech-only data, and speech-specific cepstral countermeasures (CQCC-GMM, LCNN, x-vector). The idea is sound; the artifacts are not there.
- EfficientAT has **no official ONNX or TFLite export**, despite secondary sources implying otherwise, and **no HF Hub port**. Weights come only from its own GitHub Releases. The arXiv 2509.14049 authors converted to ONNX themselves and released nothing.
- EfficientAT's README **does not document `mn01`/`mn02`** despite those names appearing in papers — smallest *documented* AudioSet models are `mn04_as` (0.983 M) and `dymn04_as` (1.97 M). (The `mn01`/`mn02` **files** do exist in the release; the README just doesn't describe them.)
- Per-class clip counts for Doorbell / Smoke detector / Glass in AudioSet are **not published anywhere** — count them from the segment TSVs in Zenodo 7096702 yourself.
- **Nonspeech100** (Hu & Wang 2010): cited constantly, **no download page, mirror or repo anywhere.** DEMAND + MUSAN noise + FSDnoisy18k cover the same need with real URLs and stated licenses.
- **Baby Chillanto (INAOE)**: no public download; signed institutional agreement only.
- `github.com/Arindam-Jain/Audio-Classification` — README advertises 1045 glass-breaking / 1350 baby-crying / 1700 noise clips. The repo contains **only code and precomputed `.npy` features**. No audio, no data source, no LICENSE. Commercial work for Hero Electronix. Sibling `shivansh2502/Audio-Classification` is the same project.
- `SoundabilityLab/SoundWatch` — the TFLite model and label files are **not in the repo**; they are behind a **Dropbox link** in the README, and the README never enumerates the classes. **Mirror it immediately if used** — single point of failure. The 31-hour / 19-class home-sounds corpus described in the CHI 2020 paper has **no downloadable link anywhere**; contact the authors.
- `docs.edgeimpulse.com/datasets/audio/...` "Glass breaking" resolves to a generic index with no download link, sample rate, or license. Use the Studio project instead.

**Miscellaneous corrections**
- Most HF "baby cry" models (`foduucom/baby-cry-classification`, `Wiam/baby-cry-classification-finetuned-babycry-v4`) classify the **reason** for crying from tiny donateacry-derived sets. Wrong task. `CAPYLEE/CRSTC` is the exception.
- **The official DCASE Task 1 baselines contain NO knowledge distillation.** Both 2024 and 2025 were fetched and confirmed: freq-mixstyle + masking + (2025) device-specific fine-tuning only, no teachers, no logits. **The KD machinery lives in the team submission repos** (`fschmid56/cpjku_dcase23`, `yqcai888/easy_dcase_task1`).
- `nttcslab/dcase2025_task4_baseline` is **not** a semi-supervised SED baseline — DCASE 2025 Task 4 changed to "Spatial Semantic Segmentation of Sound Scenes." For mean-teacher, use the 2023/2024 recipes under `DCASE-REPO/DESED_task`.
- **Git LFS trap, repeatedly:** `.tflite`/`.pb`/`.h5` in Arm ML-zoo and stm32ai-modelzoo are **131-byte pointer files** in-tree. A plain `git clone` without git-lfs, or a `wget` on `raw.githubusercontent.com`, silently yields the pointer and the flatbuffer parser fails on garbage. Use `media.githubusercontent.com/media/...` or `git lfs pull`. (ST's `.keras`/`.h5` backbones in the **services** repo are plain git — no LFS.)
- **Apple's `version1` weights are not extractable.** Searched `/System/Library` and the framework Resources dir — no browsable `.mlmodelc`; it is embedded in the framework binary.
- MicIRP is a **Blogspot site with per-microphone posts**, not a packaged dataset, and the IRs are **vintage studio microphones**, not MEMS. Measure our own INMP441 IR.
- Picovoice has **no non-speech sound event detection product**. Xperi/DTS, Bosch, Cyberon and Fluent.ai publish no class list, weights, benchmarks or ontology. Amazon Alexa Guard publishes **no accuracy figures** and has been discontinued into paid Emergency Assist.
- Sensory's **product** page publishes nothing. The useful figures are only on their **engineering blog post**.

---

## 10. Unverified leads — fenced off

Everything here is **[F]** or **[S]**: nobody adversarially re-checked it. Do not build a plan on any of it without verifying first. The most load-bearing ones are flagged.

**Class lists we could not read (verify with one grep before planning around them)**
- **VGGSound's 310 classes** — https://www.robots.ox.ac.uk/~vgg/data/vggsound/ (repo https://github.com/hche11/VGGSound). **This is the highest-value ten-minute check in the report.** It is CC BY 4.0 with the VGG page stating commercial use is allowed, >210 k 10-second clips. If its 310 classes include doorbell / alarm / crying, it is the **largest commercially-usable source we could mine, at roughly 500+ clips per class — enough on its own to fix our thin classes.** The class-list CSV 404'd. **Download `vggsound.csv` and grep it.** Audio must be scraped from YouTube.
- GISE-51's literal 51 class names — read `meta/lbl_map.csv`.
- Zenodo 3689288's 24 "pattern sound" alarm class names — read the zip's folder names.
- CochlScene's 13 class names — read `Data_info.tsv`. Verify a home/living-room class exists before treating it as a pretraining source.
- PretrainedSED's exact 447-class list — the README does not itemize it.
- BEATs' fine-tuned head class count — confirm 527 before index-mapping.
- Audio-MAE's head class count — same.
- E-PANNs — confirm the output dimension survives pruning.
- HTS-AT's AudioSet head class count.
- SoundWatch's label list (it is in the Dropbox files, not the README).
- ProtoSound's 10 categories (list the assets directory).

**Licenses never fetched — assume nothing**
HTS-AT · OpenL3 · `Qualcomm-AI-research/bcresnet` · `gveres/donateacry-corpus` · `theMoro/DIRAugmentation` · `kkoutini/cpjku_dcase20` · `marmoi/dcase2021_task1a_baseline` · `marmoi/dcase2022_task1_baseline` · `toni-heittola/dcase2020_task1_baseline` · `CPJKU/cpjku_dcase24` · TF-SepNet reference impl · DEMAND's exact CC variant · WHAM! noise-only release · DiPCo (search results said CDLA-Permissive) · sherpa-onnx model artifacts · `litert-community/PANNs-CNN14-AudioSet-LiteRT` · E-PANNs Zenodo record · `gentonesv4b.zip` · MIVIA (publishes **no terms at all**).

**Numbers quoted from secondary sources, not the primary artifact**
- TUT Rare Sound Events isolated-event counts (148 / 139 / 187) — from an arXiv summary, not the Zenodo listing.
- MIVIA's 6000 events — **counts SNR/background variants**; unique source recordings are fewer. Ask them.
- T3 timing (3 × 0.5 s pulses, 0.5 s gaps, 1.5 s pause) — secondary sources; the primary standards pages returned 403. Corroborated only empirically by Apple's classifier scoring a synthetic version at 0.994–1.000 `smoke_detector`.
- Silero VAD size — "reports vary between <1 MB and 2 MB across versions."
- Edge Impulse docs "Glass breaking" dataset: 500 items / BSD-3-Clause-Clear — from **search-result text**, not the page body.

**Leads worth a look but not chased**
- `HTS-AT` ships a **DESED sound-event-detection checkpoint** — the closest published task to ours. Worth a second look **if** its license clears.
- `Cnn14_16k_mAP=0.438.pth` (PANNs) is the only teacher trained **natively at 16 kHz**, and PANNs `MobileNetV1` is the only AudioSet model whose op set is already inside our layer table. Both worth verifying properly.
- "Creating a Good Teacher for Knowledge Distillation in Acoustic Scene Classification" (Morocutti/Schmid/Koutini/Widmer, arXiv 2503.11363) — the ablation on teacher architecture/size/ensembling is exactly the guidance we want for picking a teacher. **No public code repo found. Paper only.**
- `CPJKU/cpjku_dcase24` (1st place DCASE 2024 Task 4) — https://github.com/CPJKU/cpjku_dcase24 — publishes **pseudo-labels and checkpoints in its GitHub release** and its recipe is AudioSet-strong pretraining + **iterative self-training** (ensemble pseudo-label → retrain → repeat). A concrete template for the "big teacher expands thin classes" loop. **License unknown.**
- `DCASE-REPO/DESED_task` (**MIT**) — the canonical mean-teacher semi-supervised recipe (EMA teacher, consistency loss on unlabeled clips, dataset-specific mixup, per-class loss masking for partially-annotated data). 2024 baseline is CRNN + frozen BEATs 768-d frame embeddings; dev-test PSDS-1 0.48 ± 0.003. This is exactly our situation — a few strong labels, a pile of weak labels, a mountain of unlabeled audio.
- `litert-community/PANNs-CNN14-AudioSet-LiteRT` — https://huggingface.co/litert-community/PANNs-CNN14-AudioSet-LiteRT. License and size unknown. Not a deployment target (we don't run TFLite), but a convenient **already-quantized reference** for checking that int8 quantization of an AudioSet CNN does not destroy the `Television` / `Fire alarm` / `Glass` logits before we commit to distilling into our own int8 export.
- **Krishnamoorthi, "Quantizing deep convolutional networks for efficient inference"** — https://arxiv.org/abs/1806.08342. Relevant conclusion: **per-channel int8 weights with float activations is the safe configuration, not a trap.** Per-channel weights + per-layer 8-bit activations lands within ~2% of float post-training; QAT closes it to ~1%. **Our scheme is strictly easier than that.** So quantization is almost certainly not what causes `fire_alarm` 0.13 — spend effort on data and distillation, and only add QAT if we later quantize activations too.
- `Arm-Examples/ML-zoo` also contains **RNNoise** under `models/noise_suppression/RNNoise` (confirmed present).
- `ESC-10` (the CC-BY subset of ESC-50) is attribution-only and therefore shippable, unlike ESC-50 proper — worth checking whether the `crying_baby` clips fall inside it.
