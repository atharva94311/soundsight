// ============================================================================
//  VISUAL ALERT SYSTEM — 3D Digital Twin
//  config.js — every number the simulation reads lives here.
// ============================================================================

export const APT = {
  minX: -7, maxX: 7,
  minZ: -5, maxZ: 5,
  wallH: 1.75,
  wallT: 0.16,
};

export const ROOMS = [
  { id: 'corridor', name: 'Entry / Corridor', x0: -7,   x1: -4, z0: -5,   z1: 5,  floor: 0x1e2530 },
  { id: 'kitchen',  name: 'Kitchen',          x0: -4,   x1: 1,  z0: -5,   z1: -0.6, floor: 0x1d2a28 },
  { id: 'bedroom',  name: 'Bedroom',          x0: 1,    x1: 7,  z0: -5,   z1: -0.6, floor: 0x272334 },
  { id: 'living',   name: 'Living Room',      x0: -4,   x1: 7,  z0: -0.6, z1: 5,  floor: 0x202738 },
];

// Axis-aligned wall segments: [x0, z0, x1, z1, exterior?]
export const WALLS = [
  // ---- exterior ----
  [-7, -5,  7, -5, true],              // north
  [-7,  5,  7,  5, true],              // south
  [ 7, -5,  7,  5, true],              // east
  [-7, -5, -7, -0.75, true],           // west (above main door)
  [-7, 0.75, -7, 5, true],             // west (below main door)
  // ---- interior: corridor / rest, with two doorways ----
  [-4, -5, -4, -3.6, false],
  [-4, -2.2, -4, 1.6, false],
  [-4, 3.0, -4, 5, false],
  // ---- interior: north rooms / living, with two doorways ----
  [-4, -0.6, -0.4, -0.6, false],
  [ 0.8, -0.6, 2.4, -0.6, false],
  [ 3.8, -0.6, 7, -0.6, false],
  // ---- interior: kitchen / bedroom divider ----
  [ 1, -5, 1, -0.6, false],
];

// ---------------------------------------------------------------------------
//  Spectral profiles — 24 bins spanning 0–8 kHz (Nyquist of a 16 kHz I2S feed)
// ---------------------------------------------------------------------------
const BINS = 24;
const NYQ = 8000;

// MFCC uses a mel filterbank, so space the display bins on the mel scale too —
// it is both what the model actually sees and far more readable at low frequency.
const mel = (f) => 2595 * Math.log10(1 + f / 700);
const melInv = (m) => 700 * (10 ** (m / 2595) - 1);
const M0 = mel(80), M1 = mel(NYQ);
const binHz = (i) => melInv(M0 + ((i + 0.5) / BINS) * (M1 - M0));

/** Sum of gaussian formants -> a plausible magnitude spectrum. */
function spectrum(peaks) {
  const out = [];
  for (let i = 0; i < BINS; i++) {
    const f = binHz(i);
    let v = 0.03;
    for (const [cf, amp, width] of peaks) {
      v += amp * Math.exp(-((f - cf) ** 2) / (2 * width * width));
    }
    out.push(Math.min(1, v));
  }
  return out;
}

// ---------------------------------------------------------------------------
//  Sound events the system is trained to care about (plus one it must reject)
// ---------------------------------------------------------------------------
export const EVENTS = [
  {
    id: 'doorbell',
    label: 'Doorbell',
    key: '1',
    color: 0x3b82f6,
    css: '#3b82f6',
    priority: 'LOW',
    icon: '🔔',
    source: [-7.55, 1.15, 0.85],
    where: 'Main Door',
    signature: 'Two-tone chime · 660 Hz + 1.32 kHz · 1.2 s decay',
    mfcc: spectrum([[660, 0.95, 200], [1320, 0.7, 260], [2640, 0.28, 420]]),
    confidence: 0.96,
    oled: ['VISITOR', 'AT MAIN DOOR'],
    ledPattern: '2 slow blue pulses',
    vibration: '2 × 300 ms buzz',
    flood: false,
    escalate: false,
    phone: { title: 'Visitor at the Door', body: 'Doorbell detected at Main Door', tone: 'low' },
  },
  {
    id: 'fire',
    label: 'Fire Alarm',
    key: '2',
    color: 0xef4444,
    css: '#ef4444',
    priority: 'CRITICAL',
    icon: '🔥',
    source: [-2.9, 2.05, -3.5],
    where: 'Kitchen',
    signature: 'ISO 8201 T-3 pattern · 3.1 kHz carrier · 0.5 s on / 0.5 s off ×3',
    mfcc: spectrum([[3150, 1.0, 260], [6300, 0.45, 380], [1600, 0.15, 300]]),
    confidence: 0.98,
    oled: ['** FIRE **', 'KITCHEN — GET OUT'],
    ledPattern: 'Red strobe, 4 Hz, whole room',
    vibration: 'Continuous · bed shaker ON',
    flood: true,
    escalate: true,
    phone: { title: 'FIRE ALARM — Kitchen', body: 'Evacuate. Acknowledge within 30 s.', tone: 'critical' },
  },
  {
    id: 'cooker',
    label: 'Pressure Cooker',
    key: '3',
    color: 0xf59e0b,
    css: '#f59e0b',
    priority: 'MEDIUM',
    icon: '🍲',
    source: [-2.6, 1.15, -4.5],
    where: 'Kitchen — Stove',
    signature: 'Narrowband whistle · 2.2 kHz + harmonic · sustained',
    mfcc: spectrum([[2200, 0.92, 150], [4400, 0.5, 200], [6600, 0.2, 240]]),
    confidence: 0.91,
    oled: ['COOKER', 'WHISTLING'],
    ledPattern: '3 amber pulses',
    vibration: '3 × 200 ms buzz',
    flood: false,
    escalate: false,
    phone: { title: 'Pressure Cooker Whistling', body: 'Stove in Kitchen needs attention', tone: 'medium' },
  },
  {
    id: 'baby',
    label: 'Baby Crying',
    key: '4',
    color: 0xa855f7,
    css: '#a855f7',
    priority: 'MEDIUM',
    icon: '👶',
    source: [2.1, 0.75, 3.4],
    where: 'Living Room',
    signature: 'Harmonic stack · 450 Hz f0 · rising-falling contour',
    mfcc: spectrum([[450, 0.9, 160], [900, 0.72, 200], [1800, 0.45, 280], [2700, 0.22, 320]]),
    confidence: 0.89,
    oled: ['BABY CRYING', 'LIVING ROOM'],
    ledPattern: 'Slow violet breathing',
    vibration: '2 × 500 ms buzz',
    flood: false,
    escalate: false,
    phone: { title: 'Baby Crying', body: 'Detected in Living Room', tone: 'medium' },
  },
  {
    id: 'glass',
    label: 'Glass Breaking',
    key: '5',
    color: 0x22d3ee,
    css: '#22d3ee',
    priority: 'HIGH',
    icon: '🪟',
    source: [-3.1, 0.35, -2.1],
    where: 'Kitchen',
    signature: 'Broadband transient · 2–8 kHz · <80 ms attack',
    mfcc: spectrum([[5200, 0.85, 1900], [7200, 0.6, 900]]),
    confidence: 0.87,
    oled: ['GLASS BROKE', 'KITCHEN'],
    ledPattern: 'Cyan double-strobe',
    vibration: '4 × 200 ms buzz',
    flood: false,
    escalate: false,
    phone: { title: 'Glass Breaking', body: 'Possible hazard in Kitchen', tone: 'high' },
  },
  {
    id: 'tv',
    label: 'TV / Music',
    key: '6',
    color: 0x64748b,
    css: '#64748b',
    priority: 'IGNORE',
    icon: '📺',
    source: [6.6, 1.25, 2.2],
    where: 'Living Room',
    signature: 'Broadband speech + music · no stable signature',
    mfcc: spectrum([[300, 0.55, 260], [800, 0.48, 500], [1900, 0.3, 900], [3800, 0.14, 1200]]),
    confidence: 0.31,
    rejected: true,
    oled: null,
    ledPattern: '—',
    vibration: '—',
    flood: false,
    escalate: false,
    phone: null,
  },
];

export const EVENT_BY_ID = Object.fromEntries(EVENTS.map((e) => [e.id, e]));

// ---------------------------------------------------------------------------
//  Modules — the "building blocks" of the system
// ---------------------------------------------------------------------------
export const DEVICES = [
  {
    id: 'node-door',
    kind: 'node',
    name: 'Door Module',
    room: 'corridor',
    pos: [-6.83, 1.62, 0.9],
    yaw: Math.PI / 2,
    range: 6.0,
    classes: ['doorbell', 'glass', 'fire'],
    mcu: 'ESP32-S3 · INMP441',
    blurb: 'Wall-mounted beside the entry. Owns doorbell detection and doubles as a fallback fire listener for the corridor.',
    how: "The INMP441 is a digital MEMS microphone: it puts 24-bit samples straight onto the I\u00b2S bus, so there is no analogue preamp stage and no analogue noise to fight. Firmware fills a 1024-sample window at 16 kHz (64 ms), reduces it to 13 MFCCs \u00d7 49 frames, and runs that through a small CNN held in flash. Only the verdict \u2014 about 40 bytes \u2014 ever leaves the module. Audio never touches the network.",
    wiring: [
      ['I\u00b2S bit clock', 'INMP441 SCK \u2192 GPIO 4'],
      ['I\u00b2S word select', 'INMP441 WS \u2192 GPIO 5'],
      ['I\u00b2S data in', 'INMP441 SD \u2192 GPIO 6'],
      ['Channel select', 'INMP441 L/R \u2192 GND (picks left)'],
      ['Status LED', 'WS2812 DIN \u2192 GPIO 7, 330 \u03a9 in series'],
      ['Mic supply', 'INMP441 VDD \u2192 3V3'],
      ['Battery path', '18650 \u2192 TP4056 \u2192 MT3608 boost \u2192 5 V pin'],
    ],
    gotcha: "A 5 V WS2812 wants at least 3.5 V on its data line and the ESP32 drives 3.3 V. Run the LED from the cell (3.0\u20134.2 V) so 3.3 V is comfortably in spec, or add a 74AHCT125 buffer if you see the first pixel flicker.",
    duty: 'Listening continuously \u00b7 ~80 mA \u00b7 roughly 14 h per 2600 mAh charge',
    bom: [
      ['ESP32-S3 DevKitC', 650], ['INMP441 I²S mic', 180], ['WS2812 status LED', 25],
      ['18650 + TP4056 + MT3608 boost', 235], ['Enclosure + magnet mount', 80],
    ],
  },
  {
    id: 'node-kitchen',
    kind: 'node',
    name: 'Kitchen Module',
    room: 'kitchen',
    pos: [-1.6, 1.66, -4.84],
    yaw: 0,
    range: 6.5,
    classes: ['fire', 'cooker', 'glass'],
    mcu: 'ESP32-S3 · INMP441',
    blurb: 'Highest-risk room. Carries the extra pressure-cooker and glass profiles on top of the core fire model.',
    how: "Identical firmware to the other nodes \u2014 the difference is which classes are compiled in. The kitchen carries the pressure-cooker and glass-break profiles on top of doorbell/fire/unknown, because those sounds only ever originate here. Keeping the class list per-room shrinks the model and cuts false positives elsewhere in the flat.",
    wiring: [
      ['I\u00b2S bit clock', 'INMP441 SCK \u2192 GPIO 4'],
      ['I\u00b2S word select', 'INMP441 WS \u2192 GPIO 5'],
      ['I\u00b2S data in', 'INMP441 SD \u2192 GPIO 6'],
      ['Channel select', 'INMP441 L/R \u2192 GND (picks left)'],
      ['Status LED', 'WS2812 DIN \u2192 GPIO 7, 330 \u03a9 in series'],
      ['Mic supply', 'INMP441 VDD \u2192 3V3'],
      ['Battery path', '18650 \u2192 TP4056 \u2192 MT3608 boost \u2192 5 V pin'],
    ],
    gotcha: "Mount it away from the extractor fan. Broadband fan noise raises the floor across exactly the 2\u20134 kHz band the fire alarm lives in, which is the one place you cannot afford lost sensitivity.",
    duty: 'Listening continuously \u00b7 ~80 mA \u00b7 roughly 14 h per 2600 mAh charge',
    bom: [
      ['ESP32-S3 DevKitC', 650], ['INMP441 I²S mic', 180], ['WS2812 status LED', 25],
      ['18650 + TP4056 + MT3608 boost', 235], ['Enclosure + magnet mount', 80],
    ],
  },
  {
    id: 'node-living',
    kind: 'node',
    name: 'Living Room Module',
    room: 'living',
    pos: [3.0, 1.66, 4.84],
    yaw: Math.PI,
    range: 6.5,
    classes: ['baby', 'glass', 'doorbell', 'fire', 'tv'],
    mcu: 'ESP32-S3 · INMP441',
    blurb: 'Shared space listener. Also the node that proves the model rejects TV and music instead of crying wolf.',
    how: "The shared space is where false positives get born, so this node carries the widest class list including an explicit 'background' class trained on TV, music and conversation. Anything that lands under the 75% gate is logged and dropped, never forwarded. A system that alerts on the television gets unplugged within a week.",
    wiring: [
      ['I\u00b2S bit clock', 'INMP441 SCK \u2192 GPIO 4'],
      ['I\u00b2S word select', 'INMP441 WS \u2192 GPIO 5'],
      ['I\u00b2S data in', 'INMP441 SD \u2192 GPIO 6'],
      ['Channel select', 'INMP441 L/R \u2192 GND (picks left)'],
      ['Status LED', 'WS2812 DIN \u2192 GPIO 7, 330 \u03a9 in series'],
      ['Mic supply', 'INMP441 VDD \u2192 3V3'],
      ['Battery path', '18650 \u2192 TP4056 \u2192 MT3608 boost \u2192 5 V pin'],
    ],
    gotcha: "Train the background class on the user's own television at their own volume. A model trained on generic noise clips will still alert on their TV, because that is the one sound it never heard.",
    duty: 'Listening continuously \u00b7 ~80 mA \u00b7 roughly 14 h per 2600 mAh charge',
    bom: [
      ['ESP32-S3 DevKitC', 650], ['INMP441 I²S mic', 180], ['WS2812 status LED', 25],
      ['18650 + TP4056 + MT3608 boost', 235], ['Enclosure + magnet mount', 80],
    ],
  },
  {
    id: 'beacon',
    kind: 'beacon',
    name: 'Alert Beacon',
    room: 'bedroom',
    pos: [3.55, 0.55, -4.22],
    yaw: 0.9,
    range: 5.5,
    classes: ['fire', 'baby', 'glass'],
    mcu: 'ESP32 · WS2812 ring · SSD1306',
    blurb: 'The output hub at the bedside: 16-LED ring, OLED text, and the driver for the bed shaker. Also listens, so the bedroom is covered even if every other node dies.',
    how: "Subscribes to home/+/event, holds the priority table, and owns the 30-second acknowledgement timer. It drives three output channels at once: the 16-LED ring, the OLED line of text, and the MOSFET that runs the bed shaker. If the broker stops answering it accepts ESP-NOW frames straight from the nodes and makes the priority decision itself, so the bedroom keeps working with the hub switched off.",
    wiring: [
      ['Ring data', 'WS2812 ring DIN \u2192 GPIO 18, 330 \u03a9'],
      ['OLED data', 'SSD1306 SDA \u2192 GPIO 21'],
      ['OLED clock', 'SSD1306 SCL \u2192 GPIO 22 (I\u00b2C, addr 0x3C)'],
      ['Onboard motor', 'MOSFET gate \u2192 GPIO 19, 100 \u03a9 + 10 k\u03a9 pull-down'],
      ['Bed shaker', 'MOSFET gate \u2192 GPIO 23, screw terminal out'],
      ['Motor supply', 'ERM + \u2192 cell +, \u2212 \u2192 MOSFET drain'],
      ['Flyback', '1N4148 across each motor, cathode to +'],
      ['Battery path', '18650 \u2192 TP4056 \u2192 MT3608 boost \u2192 5 V pin'],
    ],
    gotcha: "Never hang a motor off a GPIO. An ERM pulls about 90 mA running and more at stall, against a 40 mA pin limit \u2014 and without the flyback diode the inductive kick will kill the MCU pin, usually after it has already shipped.",
    duty: 'Idle 45 mA \u00b7 full alert (ring + both motors) ~430 mA',
    bom: [
      ['ESP32 DevKit + MOSFET driver', 375], ['WS2812 ring ×16', 220], ['SSD1306 0.96" OLED', 190],
      ['ERM vibration motor', 60], ['18650 + TP4056 + MT3608 boost', 235], ['Enclosure', 80],
    ],
  },
  {
    id: 'shaker',
    kind: 'shaker',
    name: 'Bed Shaker',
    room: 'bedroom',
    pos: [5.02, 0.591, -4.5],
    yaw: 0,
    range: 0,
    classes: [],
    mcu: 'ERM motor · driven by Beacon',
    blurb: 'Slides under the pillow. This is the channel that works when hearing aids are out and eyes are shut — the single most important output at night.',
    how: "Deliberately the dumbest module in the system: a motor in a fabric pouch on a two-core lead back to the beacon's second MOSFET channel. No radio, no battery, no firmware, nothing to pair or charge. It cannot fail independently \u2014 if the beacon has power, the shaker works. That matters, because at night with hearing aids out this is the only channel that reaches the user at all.",
    wiring: [
      ['Drive +', 'Motor + \u2192 cell + at the beacon'],
      ['Drive \u2212', 'Motor \u2212 \u2192 MOSFET drain (GPIO 23)'],
      ['Flyback', '1N4148 across the motor, cathode to +'],
      ['Cable', '2-core, 1.5 m, under the mattress to the pillow'],
    ],
    gotcha: "Put it under the pillow, not under the mattress. Through a mattress the vibration is damped to nothing; against the pillow it couples to the head and reliably wakes a sleeping person.",
    duty: '~90 mA at 3.7 V while pulsing \u00b7 drawn from the beacon cell',
    bom: [['ERM vibration motor', 120], ['MOSFET + flyback diode', 30], ['Fabric pouch', 40]],
  },
  {
    id: 'band',
    kind: 'band',
    name: 'Wearable Band',
    room: 'bedroom',
    pos: [3.48, 0.564, -3.82],
    yaw: -0.5,
    range: 0,
    classes: [],
    mcu: 'ESP32-C3 · OLED · ERM',
    blurb: 'Optional module. Carries the alert with you between rooms, with the event text on a 0.96" OLED so you know what happened, not just that something did.',
    how: "An ESP32-C3 in pure ESP-NOW mode \u2014 it never associates with the Wi-Fi network, so it keeps working during an outage and pairs by MAC address rather than a password. The 0.96\u2033 OLED carries the event text, which is the whole point: a buzz alone tells you something happened, the screen tells you it was the fire alarm in the kitchen.",
    wiring: [
      ['OLED data', 'SSD1306 SDA \u2192 GPIO 5'],
      ['OLED clock', 'SSD1306 SCL \u2192 GPIO 6 (I\u00b2C, addr 0x3C)'],
      ['Motor drive', 'MOSFET gate \u2192 GPIO 3, 100 \u03a9 + 10 k\u03a9 pull-down'],
      ['Flyback', '1N4148 across the motor, cathode to +'],
      ['Battery path', '400 mAh LiPo \u2192 TP4056 \u2192 onboard 3V3 LDO'],
    ],
    gotcha: "The radio has to stay in receive to catch an unscheduled alert, and that \u2014 not the screen \u2014 sets the battery life. Roughly a day per charge, so it is an optional extra, never the primary channel.",
    duty: 'RX always on \u00b7 ~22 mA average \u00b7 about 18 h per charge',
    bom: [['ESP32-C3 Super Mini', 280], ['0.96" OLED', 190], ['ERM motor', 60], ['400 mAh LiPo', 150], ['Strap + case', 90]],
  },
  {
    id: 'phone',
    kind: 'phone',
    name: 'Phone App',
    room: 'bedroom',
    pos: [4.14, 0.5545, -3.92],
    yaw: 0.35,
    range: 0,
    classes: [],
    mcu: 'Android · MQTT over TLS',
    blurb: 'Acknowledgement, history and per-user settings. Never the only alert path — the hardware alerts stand alone if the phone is off.',
    how: "A thin client over MQTT/TLS. It carries acknowledgement, history and per-user settings, and nothing else \u2014 by design it is never the only path to the user. If the phone is dead, silenced, left in another room, or the Wi-Fi is down, the beacon, band and shaker are completely unaffected. Treating the phone as an accessory rather than the system is what makes the whole thing safe to rely on.",
    wiringTitle: 'MQTT topics',
    wiring: [
      ['Subscribe', 'home/alert \u2014 raised alerts'],
      ['Subscribe', 'home/+/event \u2014 raw node verdicts'],
      ['Publish', 'home/ack \u2014 acknowledgement'],
      ['Transport', 'MQTT over TLS, port 8883'],
    ],
    gotcha: "Android will happily kill a backgrounded MQTT client to save battery. The acknowledgement has to be reachable from the beacon's own button too, or an escalation fires while the user is looking straight at a phone that never buzzed.",
    duty: 'Push-driven \u00b7 no polling \u00b7 negligible battery cost',
    bom: [['Existing phone', 0]],
  },
  {
    id: 'hub',
    kind: 'hub',
    name: 'Hub · Decision Engine',
    room: 'living',
    pos: [-3.42, 0.755, 0.62],
    yaw: Math.PI / 2,
    range: 0,
    classes: [],
    mcu: 'Mosquitto MQTT + Python',
    blurb: 'Assigns priority, de-duplicates nodes that heard the same event, and runs the 30-second acknowledgement timer. Nodes fall back to direct ESP-NOW if it disappears.',
    how: "Mosquitto plus a short Python service. It de-duplicates events \u2014 when two nodes hear the same fire alarm within 400 ms it raises one alert, not two \u2014 assigns priority from the class, publishes to home/alert, and runs the escalation timer. It also writes the history that the phone's timeline reads. Worth being clear: this is an optimisation, not a dependency. Pull its plug and the nodes fall back to ESP-NOW and the beacon takes over the decision.",
    wiringTitle: 'Services and topics',
    wiring: [
      ['Broker', 'mosquitto :1883 \u00b7 :8883 TLS'],
      ['Engine', 'decision_engine.py subscribes home/+/event'],
      ['Output', 'publishes home/alert'],
      ['Ack in', 'subscribes home/ack, cancels escalation'],
      ['Store', 'SQLite \u2014 event history for the app'],
    ],
    gotcha: "De-duplication needs a window, not an exact match. Two nodes never timestamp the same alarm identically \u2014 400 ms of slack collapses them into one event without merging two genuinely separate ones.",
    duty: 'Runs on the existing laptop, or a \u20b92,200 Pi Zero 2 W',
    bom: [['Existing laptop / ₹2 200 Pi Zero 2 W', 0]],
  },
];

export const DEVICE_BY_ID = Object.fromEntries(DEVICES.map((d) => [d.id, d]));

// ---------------------------------------------------------------------------
//  Latency budget — sim seconds for the animation, real ms shown in the HUD
// ---------------------------------------------------------------------------
export const STAGES = [
  { id: 'sound',     name: 'Sound',           sim: 0.85, ms: 0,   detail: 'Acoustic propagation' },
  { id: 'capture',   name: 'Capture',         sim: 0.55, ms: 64,  detail: '1024-sample window @ 16 kHz' },
  { id: 'classify',  name: 'TinyML',          sim: 1.15, ms: 186, detail: 'log-mel 40×49 → CNN on ESP32-S3' },
  { id: 'transport', name: 'Transport',       sim: 0.75, ms: 38,  detail: 'MQTT QoS 1 → broker' },
  { id: 'decide',    name: 'Decision Engine', sim: 0.45, ms: 6,   detail: 'Priority + de-duplication' },
  { id: 'alert',     name: 'Alert Beacon',    sim: 0.35, ms: 11,  detail: 'LED · vibration · OLED' },
  { id: 'notify',    name: 'Phone',           sim: 0.30, ms: 24,  detail: 'Push to acknowledgement screen' },
];

export const OFFLINE_TRANSPORT = { ms: 12, detail: 'ESP-NOW peer-to-peer (no Wi-Fi)' };

export const ESCALATE_SECONDS = 30;   // real-world spec
export const ESCALATE_SIM = 14;       // sim seconds so a demo stays watchable

export const PRIORITY_COLOR = {
  CRITICAL: '#ef4444',
  HIGH:     '#22d3ee',
  MEDIUM:   '#f59e0b',
  LOW:      '#3b82f6',
  IGNORE:   '#64748b',
};

export const CONF_THRESHOLD = 0.75;
