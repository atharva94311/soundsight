# SoundSight — 3D digital twin

An interactive 3D model of the **visual alert system for hard-of-hearing users**:
a hostel flat with three listening modules, a bedside alert beacon, a bed shaker,
a wearable band and a phone app. Trigger a sound and watch the whole chain run —
microphone → TinyML on the ESP32 → MQTT → decision engine → light, vibration and text.

It also **listens for real**: `🎤 Listen with my mic` runs an actual trained
classifier on your microphone, and a real doorbell or smoke alarm drives the whole
chain. See [Live microphone](#live-microphone).

Everything runs offline in a browser. Three.js and the model weights are both
vendored, so there are no network calls and no build step.

## Run it

```bash
python3 -m http.server 8123 --directory vas3d
```

Then open <http://localhost:8123>. Any static server works; it must be `http://`
rather than `file://` because the app uses ES modules.

## Controls

| | |
|---|---|
| Orbit / pan / zoom | drag · right-drag · scroll |
| Trigger a sound | click it in the left panel, or press `1`–`6` |
| Listen for real | `🎤 Listen with my mic` in the left panel |
| Acknowledge | `Enter`, the phone's button, or click the 3D phone |
| Open a module | click it — see below |
| Close / back out | `Esc`, the close button, or click anywhere outside |

`▶ Run scripted demo` plays a 58-second sequence covering every path: doorbell,
fire with escalation, a suppressed TV, and a Wi-Fi outage.

## The module inspector

Clicking any module isolates it. The room is replaced by an opaque studio
backdrop, the module is re-parented into its own scene with three-point lighting,
and the parts separate. Labels sit in columns at the edges joined by leader
lines, rather than floating on the parts where they collided with each other.

The camera framing is measured, not hand-tuned: `framingSphere()` puts the parts
at full explode, unions their world bounding boxes, and derives the distance from
that — so a tall assembly like the beacon's LED ring is framed as correctly as a
flat one like the node.

Each module's panel carries four things:

- **How it works** — the operating principle, in plain language.
- **Connections** — the actual wiring, pin by pin (`INMP441 SD → GPIO 6`,
  `SSD1306 SCL → GPIO 22, addr 0x3C`). Software modules show topics and services
  instead.
- **Watch out** — the one real gotcha per module. These are the things that
  actually bite: a 5 V WS2812 needs ≥3.5 V logic and the ESP32 drives 3.3 V; an
  ERM motor pulls ~90 mA against a 40 mA pin limit and needs a MOSFET plus a
  flyback diode; the band's radio must stay in receive, which is what sets its
  battery life.
- **Bill of materials** — reconciled 1:1 with the labelled parts, so the callouts
  and the cost list are the same list read two ways.

The pin assignments are this build's choices, not fixed standards — change them
in `DEVICES[].wiring` in `js/config.js`. What is not arbitrary: the mic is I²S
(not analogue), the LED is a single-wire 800 kHz protocol driven off the RMT
peripheral so inference is never interrupted, the OLED is I²C at 0x3C, and the
motors are never on a bare GPIO.

## What the twin actually models

**Six sound classes, one of which must be rejected.** Doorbell, fire alarm,
pressure cooker, baby crying, glass breaking — plus TV/music, which classifies
at 31% and is suppressed by the 75% confidence gate. Demonstrating the *non*-alert
matters as much as the alerts: a system that fires on the television is unusable.

**Localisation, not just detection.** Each module carries its own class list, so
the alert reads "Fire alarm — Kitchen", named after the module that heard it
loudest. When two modules hear the same event, the decision engine de-duplicates.

**An honest latency budget.** The animation is deliberately slowed so it can be
watched, but the hub console timestamps advance at the *modelled* rate, so the log
is a real trace that sums to the quoted figure:

```
[22:47:12.000] acoustic event @ Kitchen
[22:47:12.000] Kitchen Module: capturing 1024-sample window…
[22:47:12.064] Kitchen Module: buffer full (64 ms) → inference
[22:47:12.250] MQTT ▸ home/kitchen/event  {"c":"fire","p":0.98}
[22:47:12.288] hub: Fire Alarm → priority CRITICAL (2 nodes, de-duplicated)
[22:47:12.294] beacon: Red strobe, 4 Hz, whole room · Continuous · bed shaker ON
[22:47:12.305] push ▸ phone: "FIRE ALARM — Kitchen"
```

329 ms end-to-end over MQTT; 303 ms over ESP-NOW when Wi-Fi is down.

**Failure modes, on purpose.** Turn Wi-Fi off and the nodes fall back to ESP-NOW
peer-to-peer, the beacon takes over the priority logic, and the phone shows
`OFFLINE — no push, beacon/shaker/band still alerting`. The one thing that breaks
is the one thing that *should* break, and the twin says so instead of hiding it.

**A bed-shaker policy, not a reflex.** Asleep with hearing aids out, the shaker is
the only channel that reaches the user, so CRITICAL, HIGH and MEDIUM all fire it.
LOW is held overnight — waking someone at 3 a.m. for a doorbell is its own harm.
See `shakerFires()` in `js/pipeline.js`.

**Costs that come from the parts list.** Every figure in the UI is derived from
the `bom` arrays in `js/config.js`, so the intro card and the per-module inspector
can never drift apart. Currently ₹5,630 for all six modules as built, from ₹2,520
for a hostel starter kit (one node + beacon + shaker).

## Layout

```
vas3d/
  index.html          HUD markup
  css/style.css
  js/config.js        rooms, walls, sound events, modules, BOM, latency budget
  js/world.js         builds the flat, furniture and every module (+ explode parts)
  js/pipeline.js      the state machine and all 3D effects
  js/screens.js       OLED / phone / hub / spectrum canvas renderers
  js/controls.js      orbit camera with scripted flights
  js/util.js          canvas textures, halos, projected HTML labels
  js/main.js          wiring, bloom, picking, camera presets, scripted demo
  js/audio/           the live microphone path — see below
  vendor/             three.js r170 + postprocessing (vendored, MIT)

vas3d/js/audio/
  dsp-config.js       GENERATED from ml/vas_ml/config.py — the shared constants
  mel.js              log-mel front-end; mirrors ml/vas_ml/features.py exactly
  cnn.js              the network's forward pass, written out by hand
  model-weights.js    GENERATED by ml/export.py — base64 float32 weights
  infer.js            samples → probabilities + display spectrum
  infer-worker.js     runs infer.js off the main thread
  capture-worklet.js  batches mic blocks on the audio render thread
  mic.js              getUserMedia, ring buffer, sliding 1 s window
  listener.js         gate / smoothing / refractory, and the twin event mapping
```

The two `GENERATED` files come from `ml/`; editing them by hand will be undone
the next time the model is exported.

Tuning knobs worth knowing: `EVENTS` (add a class, its signature, its alert
design), `STAGES` (the latency budget), `DEVICES` (placement, range, BOM),
`ESCALATE_SECONDS` / `ESCALATE_SIM` (real vs replayed escalation window).

## Console handle

`window.__vas` is exposed for driving the twin from devtools:

```js
__vas.trigger('fire')   // fire any event id
__vas.ack()
__vas.step(5)           // advance 5 s without waiting on the render loop
__vas.pipe.state        // full simulation state
```

## Live microphone

`🎤 Listen with my mic` switches the twin from replaying scripted events to
running a real TinyML classifier on this machine's microphone. Doorbell, fire
alarm, baby crying, glass breaking and TV/music are detected for real; the
confidence in the hub console, the number in the MQTT payload and the spectrum
strip are all the model's actual output, not the figures in `config.js`.

Audio never leaves the page — no recording, no upload, no network call. The
model is ~35k parameters, decoded from a base64 blob at load, and runs in a
worker so the 3D scene keeps its frame rate.

Train and export it first (see [../ml/README.md](../ml/README.md)):

```bash
cd ml && ./run_all.sh
```

Until you do, the button says so and the rest of the twin works exactly as
before. Browser mic processing — AGC, noise suppression, echo cancellation — is
explicitly disabled: it is tuned for speech on calls and would flatten the level
and spectral differences the model depends on.

The same weights run on the ESP32 firmware in [../firmware/esp32](../firmware/esp32),
with the same 75% gate, the same 3-of-5 smoothing and the same 8-second
refractory window, so a node behaves the way the twin says it does.

## Not modelled

**Pressure cooker is simulation-only.** Neither ESC-50 nor FSD50K has the class,
so there is nothing to train on; `cooker` still fires from the left panel and
still runs the full chain, but the microphone will never raise it.

**The latency budget is modelled, not measured.** The per-stage figures in
`config.js` are design estimates. Browser inference actually costs ~22 ms per
window on a laptop; the 186 ms in the budget is an estimate for the ESP32-S3 and
has not been measured on hardware.

**Localisation is geometric.** Which module "hears" an event is decided by
distance in the 3D scene, not by comparing real signal levels across modules.
With one microphone there is only one listener; the room attribution is the
twin's, not the model's.
