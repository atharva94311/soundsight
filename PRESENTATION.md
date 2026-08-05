# SoundSight — presentation script

Live demo: **https://atharva94311.github.io/soundsight/**
Drive the walkthrough with **→ / ←** (or a clicker). **Reset** between runs.

---

## 1. The 60-second pitch

> A deaf or hard-of-hearing person in a hostel room cannot hear the doorbell, the
> pressure cooker, a baby crying, glass breaking — or the fire alarm. Existing
> products solve one sound each, cost tens of thousands of rupees, and most of
> them stop working when the internet does.
>
> SoundSight is a set of ₹1,100 modules that listen, work out *what* the sound was
> and *which room* it came from, and convert it into light, vibration and text.
> The classification runs on the microcontroller itself — audio never leaves the
> room, so there is no privacy question and no cloud bill.
>
> This is a working 3D twin of it, and the classifier in your browser is the real
> trained model, running on your microphone.

**Open with the strongest fact:** every number on screen — 329 ms, ₹5,630 — is
computed from the parts list and the latency budget in the code. Nothing is a
mock-up.

---

## 2. How it actually works — the chain

Say this once, early, then let the demo show it.

| # | stage | what happens | cost |
|---|---|---|---|
| 1 | **Sound** | Event occurs in a room | — |
| 2 | **Capture** | INMP441 MEMS mic → I²S, 16 kHz mono, 1-second window | 64 ms |
| 3 | **TinyML** | log-mel 40×49 → 35,782-parameter CNN, on the ESP32-S3 | 186 ms |
| 4 | **Transport** | MQTT over Wi-Fi to the hub | 38 ms |
| 5 | **Decision** | priority + de-duplicate across modules | 6 ms |
| 6 | **Alert** | LED ring, OLED text, bed shaker | 11 ms |
| 7 | **Phone** | push to the acknowledgement screen | 24 ms |
| | | **total** | **329 ms** |

Three points that matter:

- **The mic never streams.** Only the verdict leaves the module — about 40 bytes,
  `{"c":"fire","p":0.98}`. That is the whole privacy argument in one line.
- **Which module heard it loudest names the room.** That is how the alert can say
  "Fire alarm — Kitchen" rather than just "Fire alarm".
- **The 30-second acknowledgement timer.** If nobody acknowledges a CRITICAL
  alert, the system assumes they *cannot* and escalates to an emergency contact.

---

## 3. The demo, beat by beat

Nine steps. Press **→** to advance. Roughly 6–8 minutes if you talk through them.

**1 · A flat that listens**
Set the scene. Three listening modules, bedside beacon, bed shaker, band, phone.
*"Everything you see is a real part with a real price. Click any module and it
comes apart with a per-component cost."*

**2 · Doorbell** — the low-priority case
*"LOW priority. Light and a buzz — but the bed shaker deliberately stays off.
Waking someone at 3 a.m. for a visitor is its own harm."*
→ This is where you show that the system has a **policy**, not just a reflex.

**3 · Fire alarm** — the critical case
*"Two modules hear it; the hub de-duplicates so you get one alert, not two. The
room floods red, the OLED says GET OUT, and the bed shaker fires — because
someone asleep with their hearing aids out cannot see a light or hear a buzzer."*

**4 · Nobody acknowledges**
*"30-second timer. No response means we assume they can't respond, and escalate."*

**5 · Acknowledged** — one tap clears everything, event goes to history.

**6 · The television is NOT an alert** ← **the most important slide**
*"This is the hard part of the problem. A system that alerts on the television
gets unplugged in a week, and then it isn't there for the fire. So the model has a
class for TV and music, and a 75% confidence gate. Watch the console: it hears it,
classifies it, and suppresses it."*
→ If you only have time for two beats, make them this one and the fire alarm.

**7 · Wi-Fi goes down**
*"The router is gone. Nodes fall back to ESP-NOW — peer-to-peer, no
infrastructure — and the beacon takes over the decision logic itself."*

**8 · Glass breaks, still offline**
*"Full alert with no internet and no hub. The phone honestly says OFFLINE — no
push. The one thing that breaks is the one thing that should break. And ESP-NOW is
actually faster: 303 ms against 329."*

**9 · Cost** — ₹5,630 for all six modules, ₹2,520 for a starter kit.

**Then the closer:** click **🎤 Listen with my mic**, play `test-sounds/fire_alarm.wav`
out loud, and let them watch the probability bars move and the alert fire in real
time. *"That is the actual trained model, running in this browser, on this
microphone."*

---

## 4. "Why do you need a hub? Can't an ESP32 do it?"

**Answer honestly: you're right, and the design already proves it.**

> Step 7 of the demo is exactly that experiment. When Wi-Fi drops, the beacon —
> which is an ESP32 — takes over priority assignment, de-duplication and the
> acknowledgement timer, and the system keeps working. So no, the hub is not
> doing anything an ESP32 cannot.

What the hub actually buys, and it is a short list:

| function | needs a hub? |
|---|---|
| Priority + de-duplication | **No** — beacon does it in fallback mode today |
| 30-second acknowledgement timer | **No** — beacon does it |
| Light / shaker / band alerts | **No** — never touched the hub |
| **Phone push notification** | **Yes** — needs an internet-reachable broker |
| **History across weeks** | Practically — an ESP32 would need an SD card |
| Multi-room config UI | Convenience only |

**So the honest position:** the hub is optional, and that is a design *strength*,
not an omission. State it that way:

> "The hub is a convenience layer, not a dependency. Most connected safety devices
> fail completely when the internet fails. This one degrades to a defined lesser
> mode — you lose the phone push, you keep every alert that actually wakes you.
> And the hub is not a product cost: it is a laptop you already own, or a ₹2,200
> Pi Zero."

If pushed further — *"then why have it at all?"* — the answer is: phone push, and
history. Both are genuinely nice, neither is safety-critical. A shipped v2 could
drop the hub entirely and have the beacon POST to a push service directly.

---

## 5. Redundancies — what I would cut, and what I would keep

Being able to critique your own design is worth more marks than defending it.

### Deliberate redundancy — keep

- **The beacon also has a microphone.** The bedroom stays covered even if every
  other module dies. This is the single most important failure case, because it is
  the one where the user is asleep.
- **Two transports** (MQTT and ESP-NOW). Different failure modes, no shared
  infrastructure.
- **Four output channels** (light, vibration, OLED text, bed shaker). A deaf user
  asleep, awake, or in another room is reachable by a different one each time.
- **Overlapping mic coverage.** This isn't waste — comparing which module hears it
  loudest is *how* the system knows which room.

### Questionable redundancy — would cut

1. **The hub as a separate always-on device.** As above: the beacon can do its job.
   Fold it in, keep a push relay.
2. **Wearable band vs phone.** Both are "carry the alert with you". The band earns
   its place only for a user who does not carry a phone or wants it silent and
   on-body. ₹770 — the first thing to cut.
3. **Three listening modules for a one-bedroom flat.** Kitchen and living room plus
   the beacon's own mic would cover it. Saves ₹1,125.

A defensible minimum is exactly the starter kit already in the code: **one node +
beacon + shaker = ₹2,520.**

---

## 6. Hard questions, with answers

**"How accurate is it?"**
Be straight. *"On public datasets it detects baby crying well, doorbell about a
third of the time, and it still false-alarms more than a shipping product should.
The reason is that it learned from ~90 strangers' doorbells, not mine. The fix is
20 minutes of recording my own doorbell and fine-tuning — that turns 'recognise
every doorbell on earth' into 'recognise this one.'"*
Owning this is far stronger than being caught out by it.

**"Why not just use a phone app?"**
A phone is asleep, face-down, on silent, in another room. And it cannot shake a
bed. The bed shaker is the entire product for the night-time case.

**"Why on-device instead of the cloud?"**
Three reasons: a microphone streaming a bedroom to a server is unacceptable; it
would stop working in a power cut or outage; and 329 ms end-to-end beats any round
trip to a data centre.

**"What if it misses a fire?"**
It is an assistive layer on top of a normal smoke alarm, not a replacement. The
smoke alarm still sounds; this makes it *visible*. Say this clearly — it removes
the safety-certification objection entirely.

**"Is the 3D thing just a mock-up?"**
No — and this is the fun answer. The same trained weights run in three places:
PyTorch for training, a hand-written JavaScript forward pass in this browser, and
hand-written C for the ESP32. There are tests that assert all three agree to
within a millionth, because a mismatch would give a model that looks confident and
is quietly wrong.

---

## 7. Numbers cheat sheet

| | |
|---|---|
| End-to-end latency | **329 ms** (MQTT) · **303 ms** (ESP-NOW) |
| Model | **35,782 parameters**, log-mel 40×49 → CNN |
| Inference | ~22 ms in browser · 4×/second |
| Firmware footprint | ~70 kB weights, 130 kB activations |
| Confidence gate | **75%**, plus 3-of-5 window agreement, 8 s refractory |
| Full system | **₹5,630** (6 modules) |
| Starter kit | **₹2,520** (1 node + beacon + shaker) |
| Sound classes | 6 — five alerts **plus TV/music, which must be rejected** |
| Training data | FSD50K + ESC-50, ~53,000 clips |

---

## 8. If the demo breaks

- Site down → run locally: `python3 -m http.server 8123 --directory vas3d`
- Mic button greyed out → needs HTTPS or localhost; the GitHub Pages link is fine
- Nothing triggers from the mic → use the left panel buttons or keys **1–6**;
  the walkthrough does not need the microphone at all
- Everything stuck → **Reset**, then **▶ Guided walkthrough**
