import * as THREE from 'three';
import { EffectComposer } from '../vendor/postprocessing/EffectComposer.js';
import { RenderPass } from '../vendor/postprocessing/RenderPass.js';
import { UnrealBloomPass } from '../vendor/postprocessing/UnrealBloomPass.js';
import { OutputPass } from '../vendor/postprocessing/OutputPass.js';

import {
  EVENTS, DEVICES, STAGES, ROOMS, OFFLINE_TRANSPORT,
  ESCALATE_SECONDS, ESCALATE_SIM,
} from './config.js';
import { TagLayer, clamp, lerp, easeOut, V } from './util.js';
import { Orbit } from './controls.js';
import { buildWorld } from './world.js';
import { Pipeline, shakerFires } from './pipeline.js';
import { drawOLED, drawBand, drawPhone, drawLaptop, drawSpectrum } from './screens.js';

// ===========================================================================
//  Renderer
// ===========================================================================
const canvas = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05070a);
scene.fog = new THREE.Fog(0x05070a, 26, 52);

const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 300);
const controls = new Orbit(camera, canvas);

// ---- lights -------------------------------------------------------------
scene.add(new THREE.HemisphereLight(0x2a3d55, 0x070a0e, 0.5));
const sun = new THREE.DirectionalLight(0x9fc0e8, 0.55);
sun.position.set(-9, 14, 8);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
sun.shadow.camera.near = 1;
sun.shadow.camera.far = 45;
const sc = sun.shadow.camera;
sc.left = -13; sc.right = 13; sc.top = 12; sc.bottom = -12;
sun.shadow.bias = -0.0012;
scene.add(sun);

// ---- world --------------------------------------------------------------
const world = buildWorld(scene);

// persistent network links (node → hub) for ambient context
const links = [];
for (const d of DEVICES) {
  if (!d.range || d.id === 'beacon') continue;
  const a = V(...d.pos);
  const b = V(...(DEVICES.find((x) => x.id === 'hub').pos)).add(V(0, 0.16, 0));
  const mid = a.clone().add(b).multiplyScalar(0.5);
  mid.y = Math.max(a.y, b.y) + a.distanceTo(b) * 0.16 + 0.3;
  const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(curve.getPoints(40)),
    new THREE.LineBasicMaterial({
      color: 0x2f6f9e, transparent: true, opacity: 0.13,
      depthWrite: false, blending: THREE.AdditiveBlending,
    }),
  );
  scene.add(line);
  links.push(line);
}
// beacon ↔ hub link
{
  const a = V(...DEVICES.find((x) => x.id === 'beacon').pos).add(V(0, 0.25, 0));
  const b = V(...DEVICES.find((x) => x.id === 'hub').pos).add(V(0, 0.16, 0));
  const mid = a.clone().add(b).multiplyScalar(0.5);
  mid.y = Math.max(a.y, b.y) + a.distanceTo(b) * 0.16 + 0.3;
  const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(curve.getPoints(40)),
    new THREE.LineBasicMaterial({
      color: 0x2f6f9e, transparent: true, opacity: 0.13,
      depthWrite: false, blending: THREE.AdditiveBlending,
    }),
  );
  scene.add(line);
  links.push(line);
}

// ---- post ---------------------------------------------------------------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.62, 0.55, 0.68);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// ===========================================================================
//  Pipeline
// ===========================================================================
const pipe = new Pipeline(scene, world, {
  onLog: (text, color) => appendLog(text, color),
  onHistory: () => {},
  onTrigger: (ev) => { setActiveEvent(ev.id); renderDesign(ev); },
  onIdle: () => { setActiveEvent(null); renderDesign(null); },
});

// ===========================================================================
//  Tags
// ===========================================================================
const tagLayer = new TagLayer(document.getElementById('tags'), camera);

for (const r of ROOMS) {
  const t = tagLayer.add(r.name, 'room');
  t.anchor.set((r.x0 + r.x1) / 2, 0.06, (r.z0 + r.z1) / 2);
}

const deviceTags = new Map();
const TAG_Y = { node: 0.3, beacon: 0.46, hub: 0.3, phone: 0.13, band: 0.12, shaker: 0.13 };
// The four bedside modules sit within centimetres of each other — fan their
// labels out in screen space so they stay readable.
const TAG_OFFSET = {
  beacon: [-60, -26], phone: [58, -4], band: [-58, 16], shaker: [46, 26],
};
for (const d of DEVICES) {
  const t = tagLayer.add(`<b>${d.name}</b>`, 'device');
  t.anchor.set(d.pos[0], d.pos[1] + (TAG_Y[d.kind] ?? 0.25), d.pos[2]);
  if (TAG_OFFSET[d.id]) t.offset = TAG_OFFSET[d.id];
  t.el.addEventListener('click', (e) => { e.stopPropagation(); inspect(d.id); });
  deviceTags.set(d.id, t);
}

// event-source tags (shown only while that event is live)
const sourceTags = new Map();
for (const ev of EVENTS) {
  const t = tagLayer.add(`${ev.icon} ${ev.label}`, 'source');
  t.anchor.set(ev.source[0], ev.source[1] + 0.3, ev.source[2]);
  t.el.style.color = ev.css;
  t.setVisible(false);
  sourceTags.set(ev.id, t);
}

// ===========================================================================
//  Isolation ("studio") pass for the exploded view
//
//  The inspected module is re-parented into its own scene with its own three-
//  point lighting, then drawn over a dark scrim on top of the composed world.
//  That is what stops the parts blending into the room behind them.
// ===========================================================================
const soloScene = new THREE.Scene();
soloScene.add(new THREE.AmbientLight(0xffffff, 0.95));
{
  const key = new THREE.DirectionalLight(0xffffff, 3.4);
  key.position.set(-4, 6, 5);
  soloScene.add(key);
  const fill = new THREE.DirectionalLight(0xa8c8ee, 1.5);
  fill.position.set(5, 1, 3);
  soloScene.add(fill);
  const rim = new THREE.DirectionalLight(0xdcebff, 2.6);   // separates dark parts
  rim.position.set(0, 3, -6);                              // from the backdrop
  soloScene.add(rim);
}

const scrimScene = new THREE.Scene();
const scrimCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const scrimMat = new THREE.MeshBasicMaterial({
  transparent: true, opacity: 0, depthTest: false, depthWrite: false, toneMapped: false,
  map: (() => {
    const c = document.createElement('canvas');
    c.width = c.height = 256;
    const g = c.getContext('2d');
    const grad = g.createRadialGradient(128, 128, 8, 128, 128, 172);
    // Fully opaque on purpose. At 93% the nearest furniture still ghosted
    // through, which is the exact "blends into the background" problem this
    // pass exists to kill. The fade-in still reads as the room dimming away.
    // Not flat black either — a soft vignette so dark components (the ESP32
    // can, the LED ring) keep something to read against.
    grad.addColorStop(0.0, 'rgb(33,44,59)');
    grad.addColorStop(0.55, 'rgb(15,21,30)');
    grad.addColorStop(1.0, 'rgb(4,6,10)');
    g.fillStyle = grad;
    g.fillRect(0, 0, 256, 256);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  })(),
});
scrimScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), scrimMat));

let soloId = null;
function soloIn(id) {
  const dev = world.devices.get(id);
  soloScene.add(dev.group);                       // world.group has an identity
  dev.group.traverse((o) => {                     // transform, so this is lossless
    if (o.isSprite) { o.userData._vis = o.visible; o.visible = false; }
  });
}
function soloOut(id) {
  const dev = world.devices.get(id);
  world.group.add(dev.group);
  dev.group.traverse((o) => {
    if (o.isSprite && o.userData._vis !== undefined) o.visible = o.userData._vis;
  });
}

// ---------------------------------------------------------------------------
//  Part callouts — labels in clean side columns joined by leader lines, rather
//  than floating on top of the parts where they collide with each other.
// ---------------------------------------------------------------------------
const SVG_NS = 'http://www.w3.org/2000/svg';
const INSPECT_PANEL_W = 358;
const calloutLayer = document.getElementById('callouts');
const leaderSvg = document.getElementById('leaders');
const callouts = [];

function clearCallouts() {
  for (const c of callouts) { c.el.remove(); c.line.remove(); c.dot.remove(); }
  callouts.length = 0;
}

function makeCallouts(dev) {
  clearCallouts();
  for (const m of dev.parts.filter((p) => p.userData.part.price > 0)) {
    const p = m.userData.part;
    const el = document.createElement('div');
    el.className = 'callout';
    el.innerHTML = `${p.name}<span class="price">₹${p.price}</span>`;
    calloutLayer.appendChild(el);
    const line = document.createElementNS(SVG_NS, 'polyline');
    const dot = document.createElementNS(SVG_NS, 'circle');
    dot.setAttribute('r', '2.5');
    leaderSvg.appendChild(line);
    leaderSvg.appendChild(dot);
    callouts.push({ el, line, dot, mesh: m });
  }
}

function placeColumn(list, area, side, fade) {
  const GAP = 30;
  let prev = -Infinity;
  for (const p of list) { p.ly = Math.max(p.y, prev + GAP); prev = p.ly; }
  const overflow = prev - area.b;
  if (overflow > 0) for (const p of list) p.ly -= overflow;

  for (const p of list) {
    p.ly = clamp(p.ly, area.t, area.b);
    const el = p.c.el;
    const w = el.offsetWidth;
    const lx = side === 'l' ? area.l : area.r - w;
    el.style.transform = `translate(${lx.toFixed(1)}px, ${(p.ly - 11).toFixed(1)}px)`;
    el.style.opacity = fade;
    const edge = side === 'l' ? lx + w + 4 : lx - 4;
    const stub = side === 'l' ? edge + 14 : edge - 14;
    p.c.line.setAttribute('points', `${edge},${p.ly} ${stub},${p.ly} ${p.x.toFixed(1)},${p.y.toFixed(1)}`);
    p.c.dot.setAttribute('cx', p.x.toFixed(1));
    p.c.dot.setAttribute('cy', p.y.toFixed(1));
    p.c.line.style.opacity = fade;
    p.c.dot.style.opacity = fade;
  }
}

function layoutCallouts(k) {
  if (!callouts.length) return;
  const W = innerWidth, H = innerHeight;
  const area = { l: 24, r: W - INSPECT_PANEL_W - 24, t: 100, b: H - 44 };
  const fade = String(clamp(k * 1.7 - 0.55, 0, 1));
  const pts = [];
  for (const c of callouts) {
    c.mesh.getWorldPosition(wp);
    wp.project(camera);
    pts.push({ c, x: (wp.x * 0.5 + 0.5) * W, y: (-wp.y * 0.5 + 0.5) * H });
  }
  const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
  placeColumn(pts.filter((p) => p.x < cx).sort((a, b) => a.y - b.y), area, 'l', fade);
  placeColumn(pts.filter((p) => p.x >= cx).sort((a, b) => a.y - b.y), area, 'r', fade);
}

/**
 * Measure the module at full explode so the camera can frame whatever it
 * actually occupies. Hand-picked radii cropped the taller assemblies.
 */
function framingSphere(dev) {
  const saved = dev.parts.map((m) => m.position.clone());
  for (const m of dev.parts) {
    if (m.userData.explode) m.position.copy(m.userData.home).add(m.userData.explode);
  }
  dev.group.updateMatrixWorld(true);

  const bb = new THREE.Box3();
  const tmp = new THREE.Box3();
  dev.group.traverse((o) => {
    if (!o.isMesh || !o.geometry) return;          // isMesh excludes the halos
    if (!o.geometry.boundingBox) o.geometry.computeBoundingBox();
    tmp.copy(o.geometry.boundingBox).applyMatrix4(o.matrixWorld);
    bb.union(tmp);
  });

  dev.parts.forEach((m, i) => m.position.copy(saved[i]));
  dev.group.updateMatrixWorld(true);

  return {
    center: bb.getCenter(new THREE.Vector3()),
    radius: bb.getSize(new THREE.Vector3()).length() * 0.5,
  };
}

/** Shift the frustum left so the model is centred beside the detail panel. */
function applyViewOffset() {
  if (inspected) {
    camera.setViewOffset(innerWidth, innerHeight, INSPECT_PANEL_W / 2, 0, innerWidth, innerHeight);
  } else {
    camera.clearViewOffset();
  }
}

// ===========================================================================
//  UI — event buttons
// ===========================================================================
const evWrap = document.getElementById('events');
for (const ev of EVENTS) {
  const b = document.createElement('button');
  b.className = 'ev';
  b.dataset.ev = ev.id;
  b.style.color = ev.css;
  b.innerHTML = `
    <span class="swatch" style="background:${ev.css}"></span>
    <span class="name" style="color:var(--ink)">${ev.icon} ${ev.label}</span>
    <span class="pri">${ev.priority}</span>
    <span class="kb">${ev.key}</span>`;
  b.addEventListener('click', () => pipe.trigger(ev.id));
  evWrap.appendChild(b);
}
function setActiveEvent(id) {
  for (const b of evWrap.children) b.classList.toggle('active', b.dataset.ev === id);
  for (const [eid, t] of sourceTags) t.setVisible(eid === id);
}

// ---- signal chain -------------------------------------------------------
const chainEl = document.getElementById('chain');
STAGES.forEach((s, i) => {
  const li = document.createElement('li');
  li.innerHTML = `<span class="n">${i + 1}</span><span class="nm">${s.name}</span><span class="ms">${s.ms ? s.ms + ' ms' : '—'}</span>`;
  chainEl.appendChild(li);
});
const ackLi = document.createElement('li');
ackLi.innerHTML = `<span class="n">✓</span><span class="nm">Acknowledge</span><span class="ms">30 s</span>`;
chainEl.appendChild(ackLi);

// ---- phone --------------------------------------------------------------
document.getElementById('phoneMount').appendChild(world.screens.phone.canvas);
document.getElementById('phoneMount').addEventListener('click', (e) => {
  const r = e.currentTarget.getBoundingClientRect();
  const y = (e.clientY - r.top) / r.height;
  if (y > 0.44 && y < 0.55) pipe.ack();
});
const ackBtn = document.getElementById('ackBtn');
ackBtn.addEventListener('click', () => pipe.ack());

// ---- toggles ------------------------------------------------------------
const flags = { wifi: true, night: true, mains: true, coverage: false, slow: false };
document.querySelectorAll('.tg').forEach((btn) => {
  btn.addEventListener('click', () => {
    const k = btn.dataset.t;
    flags[k] = !flags[k];
    btn.classList.toggle('on', flags[k]);
    applyFlags();
  });
});
function applyFlags() {
  pipe.state.wifi = flags.wifi;
  pipe.state.night = flags.night;
  pipe.state.mains = flags.mains;
  pipe.state.battery = flags.mains ? 94 : 71;
  for (const l of links) l.material.opacity = flags.wifi ? 0.13 : 0.03;
  for (const [, dev] of world.devices) {
    if (dev.coverage) for (const m of dev.coverage) m.visible = flags.coverage;
  }
  if (pipe.state.event) renderDesign(pipe.state.event);   // policy rows depend on night mode
  document.getElementById('condNote').textContent = !flags.wifi
    ? 'Offline: nodes are talking straight to the beacon over ESP-NOW and the beacon is running the priority logic. Phone push is the only thing lost.'
    : (!flags.mains
      ? 'On battery: every module has an 18650 cell. Measured idle draw gives roughly 14 hours of listening per charge.'
      : 'Wi-Fi down? Nodes fall back to ESP-NOW and the beacon runs the decision engine locally.');
}
applyFlags();

// ---- camera presets -----------------------------------------------------
const PRESETS = {
  overview: { target: V(0.4, 0.6, -0.2), theta: -0.85, phi: 0.92, radius: 17 },
  bedroom:  { target: V(4.6, 0.7, -3.5), theta: 0.66, phi: 1.14, radius: 7.4 },
  kitchen:  { target: V(-2.0, 0.95, -3.5), theta: 0.55, phi: 1.12, radius: 7.2 },
  door:     { target: V(-6.3, 1.0, 0.6), theta: -1.45, phi: 1.05, radius: 5.6 },
  living:   { target: V(2.6, 0.5, 2.8), theta: -0.62, phi: 0.78, radius: 9.0 },
  hub:      { target: V(-3.4, 0.85, 0.62), theta: -0.95, phi: 1.12, radius: 3.1 },
  top:      { target: V(0.2, 0, -0.2), theta: -0.0001, phi: 0.055, radius: 19.5 },
};
document.querySelectorAll('.cams button').forEach((b) => {
  b.addEventListener('click', () => {
    closeInspector();
    controls.flyTo(PRESETS[b.dataset.cam]);
    document.querySelectorAll('.cams button').forEach((x) => x.classList.toggle('active', x === b));
  });
});
document.querySelector('.cams button').classList.add('active');

// ===========================================================================
//  Inspector / exploded view
// ===========================================================================
let inspected = null;
let explodeK = 0;
const inspEl = document.getElementById('inspector');

function inspect(id) {
  const dev = world.devices.get(id);
  if (!dev) return;
  if (inspected === id) { closeInspector(); return; }
  if (inspected && inspected !== id) explodeK = 0;   // restart the animation
  inspected = id;
  const def = dev.def;

  document.getElementById('inspName').textContent = def.name;
  document.getElementById('inspMcu').textContent = def.mcu;
  document.getElementById('inspBlurb').textContent = def.blurb;
  document.getElementById('inspHow').textContent = def.how || '';
  document.getElementById('inspWireTitle').textContent = def.wiringTitle || 'Connections';
  document.getElementById('inspWire').innerHTML = (def.wiring || [])
    .map(([sig, conn]) => `<tr><td>${sig}</td><td>${conn}</td></tr>`).join('');
  document.getElementById('inspGotcha').textContent = def.gotcha || '';
  document.getElementById('inspDuty').textContent = def.duty || '';

  const tbl = document.getElementById('inspBom');
  tbl.innerHTML = def.bom.map(([n, p]) =>
    `<tr><td>${n}</td><td>${p ? '₹' + p.toLocaleString('en-IN') : '—'}</td></tr>`).join('');
  const total = def.bom.reduce((s, [, p]) => s + p, 0);
  document.getElementById('inspTotal').textContent = total ? '₹' + total.toLocaleString('en-IN') : 'reuses existing hardware';
  inspEl.classList.remove('hidden');
  inspEl.scrollTop = 0;
  document.body.classList.add('inspecting');

  // Callouts only for line items that carry a cost — sub-components that ship on
  // a parent board would otherwise stack a second label on the same point.
  makeCallouts(dev);
  applyViewOffset();

  // Frame the whole exploded spread, whatever shape it happens to be.
  const fit = framingSphere(dev);
  const radius = Math.max(0.3, fit.radius / Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * 1.22);
  controls.flyTo({
    target: fit.center, radius, phi: 1.06, theta: def.yaw + 0.5, duration: 1.1,
  });
  for (const t of deviceTags.values()) t.setVisible(false);
}

function closeInspector() {
  if (!inspected) return;
  inspected = null;
  inspEl.classList.add('hidden');
  document.body.classList.remove('inspecting');
  clearCallouts();
  applyViewOffset();
  for (const t of deviceTags.values()) { t.visible = true; }
}
document.getElementById('inspClose').addEventListener('click', () => {
  closeInspector();
  controls.flyTo(PRESETS.overview);
});

// ---- picking ------------------------------------------------------------
const ray = new THREE.Raycaster();
const ndc = new THREE.Vector2();
let downAt = null;
canvas.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
canvas.addEventListener('pointerup', (e) => {
  if (!downAt) return;
  const moved = Math.abs(e.clientX - downAt[0]) + Math.abs(e.clientY - downAt[1]);
  downAt = null;
  if (moved > 6) return;

  ndc.x = (e.clientX / innerWidth) * 2 - 1;
  ndc.y = -(e.clientY / innerHeight) * 2 + 1;
  ray.setFromCamera(ndc, camera);
  const hits = ray.intersectObjects([...world.devices.values()].map((d) => d.group), true);
  if (!hits.length) { closeInspector(); return; }
  let o = hits[0].object;
  while (o && !o.userData.deviceId) o = o.parent;
  if (!o) return;
  const id = o.userData.deviceId;
  // While a module is open, everything else is behind the scrim — a click out
  // there means "get me out of here", not "open that one instead".
  if (inspected && id !== inspected) { closeInspector(); return; }
  if (id === 'phone' && ['alerting', 'escalated'].includes(pipe.state.mode)) { pipe.ack(); return; }
  inspect(id);
});

// ---- keyboard -----------------------------------------------------------
addEventListener('keydown', (e) => {
  // Arrow keys step the walkthrough — the same keys a clicker sends, so the
  // demo can be driven from the back of a room without touching the laptop.
  if (typeof demo !== 'undefined' && demo) {
    if (e.key === 'ArrowRight' || e.key === 'PageDown') {
      e.preventDefault(); document.getElementById('demoNext').click(); return;
    }
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
      e.preventDefault(); document.getElementById('demoPrev').click(); return;
    }
  }
  const ev = EVENTS.find((x) => x.key === e.key);
  if (ev) { pipe.trigger(ev.id); return; }
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pipe.ack(); }
  if (e.key === 'Escape') { closeInspector(); }
});

// ===========================================================================
//  Log + design card
// ===========================================================================
const logEl = document.getElementById('log');
/**
 * Append one line as it is emitted. (Do not diff against pipe.state.console —
 * that array is ring-buffered, so its length stops growing and an index-based
 * renderer silently stops updating once it saturates.)
 */
function appendLog(text, color) {
  const d = document.createElement('div');
  d.textContent = text;
  if (color) d.style.color = color;
  logEl.appendChild(d);
  while (logEl.children.length > 150) logEl.removeChild(logEl.firstChild);
  logEl.scrollTop = logEl.scrollHeight;
}
for (const l of pipe.state.console) appendLog(l.text, l.color);

const designBody = document.getElementById('designBody');
function renderDesign(ev) {
  if (!ev) {
    designBody.className = 'design-empty';
    designBody.textContent = 'Idle — pick a sound on the left.';
    return;
  }
  designBody.className = '';
  const row = (k, v) => `<div class="drow"><span class="dk">${k}</span><span class="dv">${v}</span></div>`;
  designBody.innerHTML = `
    <div class="dhead" style="color:${ev.css}">
      <span class="swatch" style="background:${ev.css}"></span>
      <b style="color:var(--ink)">${ev.label}</b>
      <span class="pri">${ev.priority}</span>
    </div>
    ${row('Signature', ev.signature)}
    ${ev.rejected
      ? row('Outcome', 'Confidence 31% — below the 75% gate, suppressed. No alert is raised.')
      : row('Light', ev.ledPattern) +
        row('Haptics', ev.vibration) +
        row('Bed shaker', shakerFires(ev.priority, flags.night)
          ? (flags.night ? 'Fires — user is asleep, aids out' : 'Fires — critical overrides everything')
          : (flags.night ? 'Held — too low-priority to wake someone' : 'Not needed — beacon and band are in reach')) +
        row('OLED', ev.oled ? `“${ev.oled[0]} / ${ev.oled[1]}”` : '—') +
        row('Room flood', ev.flood ? 'Yes — ceiling strip + room strobe' : 'No') +
        row('Escalation', ev.escalate
          ? `${ESCALATE_SECONDS} s → warden / emergency contact <span style="color:var(--dimmer)">(replayed at ${(ESCALATE_SECONDS / ESCALATE_SIM).toFixed(1)}× here)</span>`
          : 'Not required')}
  `;
}

// ===========================================================================
//  Scripted demo
// ===========================================================================
// Each beat says what it is doing AND what to look at while it happens. The old
// version wrote one line into the hub console, which is unreadable from the back
// of a room — by the time you find it the effect has already played.
const DEMO = [
  {
    chapter: 'the idea',
    title: 'A flat that listens',
    watch: 'Three <b>listening modules</b> (entry, kitchen, living room), a bedside '
         + '<b>beacon</b>, a <b>bed shaker</b> under the pillow, a <b>wearable band</b> '
         + 'and a <b>phone</b>. Each module is an ESP32 with a microphone that classifies '
         + 'sound <em>on the device</em> — audio never leaves the room.',
    cam: 'overview', hold: 9,
  },
  {
    chapter: 'low priority',
    title: 'Someone rings the doorbell',
    watch: 'Watch the <b>Door Module</b> light up as it hears the sound, then a packet '
         + 'fly to the hub. The beacon gives <em>two slow blue pulses</em> and the band '
         + 'buzzes twice. The <b>bed shaker stays off</b> — this is LOW priority, and '
         + 'waking someone at 3 a.m. for a visitor is its own harm.',
    cam: 'door', trigger: 'doorbell', hold: 11,
  },
  {
    chapter: 'critical',
    title: 'Fire alarm in the kitchen',
    watch: 'The <b>Kitchen Module</b> hears it loudest, so the alert reads '
         + '"Fire Alarm — Kitchen". <em>Two</em> modules hear it and the hub '
         + 'de-duplicates. Now the whole room floods red at 4 Hz, the OLED reads '
         + '"** FIRE ** GET OUT", and the <b>bed shaker fires</b> — the only channel '
         + 'that reaches someone asleep with hearing aids out.',
    cam: 'kitchen', trigger: 'fire', hold: 7,
  },
  {
    chapter: 'critical',
    title: 'Nobody acknowledges',
    watch: 'Look at the phone: a <b>30-second countdown</b> is running. If the user '
         + 'does not acknowledge, the system assumes they cannot — and <em>escalates</em>, '
         + 'calling the emergency contact. Wait for the timer, or press Next.',
    cam: 'bedroom', hold: 16,
  },
  {
    chapter: 'critical',
    title: 'Acknowledged',
    watch: 'One tap on the phone (or the beacon button) clears everything — lights, '
         + 'shaker, band. The event stays in <b>history</b> so it can be reviewed later.',
    cam: 'bedroom', ack: true, hold: 7,
  },
  {
    chapter: 'the hard part',
    title: 'The television is NOT an alert',
    watch: 'This is the most important step. The model hears the TV, classifies it, '
         + 'and the confidence lands <em>below the 75% gate</em> — so it is '
         + '<b>suppressed</b>. Watch the console: "unknown … → suppressed". A system '
         + 'that cries fire at the television gets unplugged in a week.',
    cam: 'living', trigger: 'tv', hold: 10,
  },
  {
    chapter: 'failure mode',
    title: 'Wi-Fi goes down',
    watch: 'The router is gone. Nodes fall back to <b>ESP-NOW</b>, talking peer-to-peer '
         + 'straight to the beacon, which takes over the decision logic itself. '
         + 'Notice the path on screen no longer goes through the hub.',
    cam: 'overview', wifi: false, hold: 8,
  },
  {
    chapter: 'failure mode',
    title: 'Glass breaks — still alerts, offline',
    watch: 'Full alert with no internet and no hub: light, vibration, shaker, band. '
         + 'The phone shows <b>OFFLINE — no push</b>. The one thing that breaks is the '
         + 'one thing that <em>should</em> break, and it says so instead of hiding it. '
         + 'ESP-NOW is actually <em>faster</em>: 303 ms vs 329 ms.',
    cam: 'overview', trigger: 'glass', hold: 12,
  },
  {
    chapter: 'wrap up',
    title: 'Back online',
    watch: 'Wi-Fi restored, three nodes report in. Total cost <b>₹5,630</b> for all six '
         + 'modules, or <b>₹2,520</b> for a hostel starter kit — one node, a beacon and '
         + 'a shaker. Every figure comes from the parts list, so the numbers cannot drift.',
    cam: 'hub', wifi: true, hold: 9,
  },
];

const capEl = document.getElementById('caption');
const capNum = document.getElementById('capNum');
const capChapter = document.getElementById('capChapter');
const capTitle = document.getElementById('capTitle');
const capWatch = document.getElementById('capWatch');
const btnDemo = document.getElementById('autoDemo');
const btnPrev = document.getElementById('demoPrev');
const btnNext = document.getElementById('demoNext');
const btnPlay = document.getElementById('demoPlay');

let demo = null;   // { i, auto, t }

function showCaption(i) {
  const b = DEMO[i];
  capNum.textContent = `${i + 1} / ${DEMO.length}`;
  capChapter.textContent = b.chapter;
  capTitle.textContent = b.title;
  capWatch.innerHTML = b.watch;
  capEl.classList.remove('hidden');
}

/** Apply one beat: move the camera, flip conditions, fire the event. */
function runBeat(i) {
  const b = DEMO[i];
  if (!b) return;
  demo.i = i;
  showCaption(i);

  if (b.cam && PRESETS[b.cam]) {
    controls.flyTo(PRESETS[b.cam]);
    document.querySelectorAll('.cams button')
      .forEach((x) => x.classList.toggle('active', x.dataset.cam === b.cam));
  }
  if (b.wifi !== undefined) {
    flags.wifi = b.wifi;
    document.querySelector('[data-t="wifi"]').classList.toggle('on', b.wifi);
    applyFlags();
  }
  if (b.trigger) pipe.trigger(b.trigger);
  if (b.ack) pipe.ack();

  btnPrev.disabled = i === 0;
  btnNext.textContent = i === DEMO.length - 1 ? 'Finish' : 'Next ›';
  if (demo) demo.t = 0;
}

function startDemo(auto) {
  if (listener) micBtn.click();   // the mic and the script cannot both drive the pipeline
  pipe.reset();
  demo = { i: -1, auto, t: 0 };
  btnDemo.textContent = '■ Stop';
  [btnPrev, btnNext, btnPlay].forEach((b) => { b.hidden = false; });
  btnPlay.textContent = auto ? '❙❙ Pause' : '▶ Auto';
  runBeat(0);
}

function stopDemo() {
  demo = null;
  btnDemo.textContent = '▶ Guided walkthrough';
  [btnPrev, btnNext, btnPlay].forEach((b) => { b.hidden = true; });
  capEl.classList.add('hidden');
}

btnDemo.addEventListener('click', () => (demo ? stopDemo() : startDemo(false)));
btnNext.addEventListener('click', () => {
  if (!demo) return;
  if (demo.i >= DEMO.length - 1) { stopDemo(); pipe.reset(); return; }
  runBeat(demo.i + 1);
});
btnPrev.addEventListener('click', () => {
  if (!demo || demo.i <= 0) return;
  pipe.reset();
  runBeat(demo.i - 1);
});
btnPlay.addEventListener('click', () => {
  if (!demo) return;
  demo.auto = !demo.auto;
  demo.t = 0;
  btnPlay.textContent = demo.auto ? '❙❙ Pause' : '▶ Auto';
});

document.getElementById('clearBtn').addEventListener('click', () => {
  stopDemo();
  pipe.reset();
  listener?.clearHistory();
  closeInspector();
});

// ===========================================================================
//  Live microphone
//
//  The listener module is imported on demand rather than at startup, because it
//  pulls in the exported model weights. If the model has not been trained and
//  exported yet, the twin still runs exactly as before and the button explains
//  what is missing, instead of the whole page failing to load.
// ===========================================================================
const micBtn = document.getElementById('micBtn');
const liveHint = document.getElementById('liveHint');
const liveBody = document.getElementById('liveBody');
const liveMeters = document.getElementById('liveMeters');
const levelBar = document.getElementById('levelBar');
const probsEl = document.getElementById('probs');
let listener = null;
let probRows = null;

function buildProbRows(classes) {
  probsEl.innerHTML = '';
  probRows = classes.map((c) => {
    const row = document.createElement('div');
    row.className = 'prow';
    row.innerHTML = `<span>${c}</span><div class="bar"><i></i></div><b>0</b>`;
    probsEl.appendChild(row);
    return { row, bar: row.querySelector('i'), val: row.querySelector('b') };
  });
}

function liveState(mode, msg) {
  micBtn.classList.toggle('on', mode === 'on');
  micBtn.classList.toggle('err', mode === 'err');
  micBtn.textContent = mode === 'on' ? '■ Stop listening' : '🎤 Listen with my mic';
  liveHint.textContent = mode === 'on' ? 'listening' : mode === 'err' ? 'error' : 'off';
  liveMeters.classList.toggle('hidden', mode !== 'on');
  liveBody.style.display = mode === 'on' ? 'none' : '';
  if (msg) liveBody.textContent = msg;
}

async function startListening() {
  let mod;
  try {
    mod = await import('./audio/listener.js');
  } catch (err) {
    console.error(err);
    liveState('err',
      'No trained model found. Run the pipeline in ml/ (fetch_data.sh → train.py → ' +
      'export.py) to generate js/audio/model-weights.js, then reload.');
    return;
  }

  if (!mod.Listener.available()) {
    liveState('err',
      'This browser cannot capture audio here. getUserMedia needs a secure context — ' +
      'open the twin over http://localhost rather than as a file:// URL.');
    return;
  }

  // The scripted demo and the microphone both drive the pipeline; running them
  // together produces nonsense, so starting one stops the other.
  demo = null;
  document.getElementById('autoDemo').textContent = '▶ Run scripted demo';
  pipe.reset();

  listener = new mod.Listener({
    onDetect: (eventId, live) => pipe.trigger(eventId, live),
    onProbs: (probs, rms, classes) => {
      if (!probRows) buildProbRows(classes);
      // ~0.2 RMS is a loud room; scale so normal speech sits mid-meter.
      levelBar.style.width = `${Math.min(100, (rms / 0.2) * 100).toFixed(1)}%`;
      let top = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[top]) top = i;
      probRows.forEach((r, i) => {
        r.bar.style.width = `${(probs[i] * 100).toFixed(1)}%`;
        r.val.textContent = Math.round(probs[i] * 100);
        r.row.classList.toggle('hot', i === top && probs[i] >= 0.5);
      });
    },
    onError: (err) => console.error('[vas] listener', err),
  });

  try {
    await listener.start();
    liveState('on');
  } catch (err) {
    listener = null;
    const denied = err && (err.name === 'NotAllowedError' || err.name === 'SecurityError');
    liveState('err', denied
      ? 'Microphone permission denied. Allow mic access for this page and try again.'
      : `Could not start audio capture: ${err?.message || err}`);
  }
}

micBtn.addEventListener('click', async () => {
  if (listener) {
    await listener.stop();
    listener = null;
    liveState('off',
      "Runs the real TinyML classifier on this machine's microphone. Nothing is " +
      'recorded or uploaded — audio never leaves the page.');
    return;
  }
  micBtn.disabled = true;
  try { await startListening(); } finally { micBtn.disabled = false; }
});

/** Auto mode only: hold each beat for its own duration, then advance.
 *  In manual mode this does nothing and the Next button drives it. */
function stepDemo(dt) {
  if (!demo || !demo.auto) return;
  demo.t += dt;
  const b = DEMO[demo.i];
  if (!b || demo.t < (b.hold ?? 8)) return;

  if (demo.i >= DEMO.length - 1) { stopDemo(); return; }
  runBeat(demo.i + 1);
}

// ===========================================================================
//  HUD refresh
// ===========================================================================
const chipState = document.getElementById('chip-state');
const chipNet = document.getElementById('chip-net');
const chipPower = document.getElementById('chip-power');
const chipMode = document.getElementById('chip-mode');
const chipLat = document.getElementById('chip-lat');
const budgetEl = document.getElementById('budget');

const MODE_LABEL = {
  idle: ['ARMED', 'ok'], sensing: ['LISTENING', 'warn'], classifying: ['CLASSIFYING', 'warn'],
  transport: ['PUBLISHING', 'warn'], deciding: ['DECIDING', 'warn'],
  alerting: ['ALERTING', 'crit'], escalated: ['ESCALATED', 'crit'],
  acked: ['ACKNOWLEDGED', 'ok'], rejected: ['SUPPRESSED', ''],
};

function refreshHUD() {
  const st = pipe.state;
  const [label, cls] = MODE_LABEL[st.mode] ?? ['ARMED', 'ok'];
  chipState.innerHTML = `<span class="dot ${cls}"></span><b>${label}</b>`;
  if (st.event && ['alerting', 'escalated'].includes(st.mode)) {
    chipState.style.color = st.event.css;
  } else chipState.style.color = '';

  chipNet.innerHTML = `<span class="dot ${st.wifi ? 'ok' : 'warn'}"></span>${st.wifi ? 'MQTT · 3 nodes' : 'ESP-NOW direct'}`;
  chipPower.innerHTML = `<span class="dot ${st.mains ? 'ok' : 'warn'}"></span>${st.mains ? 'Mains' : 'Battery'} · ${st.battery}%`;
  chipMode.innerHTML = `<span class="dot"></span>${st.night ? 'Sleep mode' : 'Normal mode'}`;

  const total = st.totalMs || 0;
  chipLat.textContent = total ? `${total} ms` : '— ms';
  budgetEl.textContent = total ? `${total} ms` : '— ms';

  // chain
  const items = chainEl.children;
  for (let i = 0; i < STAGES.length; i++) {
    const li = items[i];
    // A suppressed event never reaches transport — don't light that row up.
    li.classList.toggle('active', st.stage === i && st.mode !== 'idle' && st.mode !== 'rejected');
    li.classList.toggle('done', st.stage > i || (st.mode === 'alerting' && i < STAGES.length));
    if (STAGES[i].id === 'transport') {
      li.querySelector('.ms').textContent = (st.wifi ? STAGES[i].ms : OFFLINE_TRANSPORT.ms) + ' ms';
      li.querySelector('.nm').textContent = st.wifi ? 'Transport · MQTT' : 'Transport · ESP-NOW';
    }
  }
  const ackItem = items[STAGES.length];
  ackItem.classList.toggle('active', st.mode === 'alerting' && st.event?.escalate);
  ackItem.classList.toggle('done', st.mode === 'acked');
  ackItem.querySelector('.ms').textContent =
    st.mode === 'alerting' && st.event?.escalate ? `${Math.ceil(st.ackLeft)} s` :
    st.mode === 'escalated' ? 'CALLING' : `${ESCALATE_SECONDS} s`;

  ackBtn.disabled = !['alerting', 'escalated'].includes(st.mode);
  ackBtn.textContent = st.mode === 'escalated' ? 'Cancel emergency call'
    : (!st.wifi ? 'Acknowledge on beacon' : 'Acknowledge');
}

// ===========================================================================
//  Loop
// ===========================================================================
function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  composer.setSize(w, h);
  bloom.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  applyViewOffset();
}
addEventListener('resize', resize);
resize();

const clock = new THREE.Clock();
let screenAcc = 0;
const wp = new THREE.Vector3();

/** One simulation step, no drawing. Shared by the render loop and __vas.step. */
function advance(raw) {
  const dt = raw * (flags.slow ? 0.28 : 1);

  stepDemo(dt);
  controls.update(raw);
  pipe.update(dt);

  // exploded view interpolation
  const wantK = inspected ? 1 : 0;
  explodeK = lerp(explodeK, wantK, raw * 6);
  for (const [id, dev] of world.devices) {
    const k = id === inspected ? explodeK : 0;
    if (k < 0.001 && !dev._wasExploded) continue;
    dev._wasExploded = k > 0.001;
    for (const m of dev.parts) {
      const ex = m.userData.explode;
      if (!ex) continue;
      m.position.copy(m.userData.home).addScaledVector(ex, easeOut(k));
    }
  }
  // move the inspected module into the studio scene, and back out once the
  // scrim has finished fading so it never pops back into a lit room
  scrimMat.opacity = lerp(scrimMat.opacity, inspected ? 1 : 0, raw * 6);
  if (inspected && soloId !== inspected) {
    if (soloId) soloOut(soloId);
    soloIn(inspected);
    soloId = inspected;
  } else if (!inspected && soloId && scrimMat.opacity < 0.012) {
    soloOut(soloId);
    soloId = null;
  }
  layoutCallouts(explodeK);

  // device tags hide while inspecting
  for (const [id, t] of deviceTags) t.visible = !inspected;

  // Spectrum panels are world-space sprites, so scale them by camera distance
  // to keep a constant on-screen size whether you are zoomed in or out.
  for (const [, dev] of world.devices) {
    if (!dev.spectrum) continue;
    dev.spectrum.sprite.getWorldPosition(wp);
    const d = camera.position.distanceTo(wp);
    const width = clamp(d * 0.17, 0.7, 3.4);
    dev.spectrum.sprite.scale.set(width, width / 3.2, 1);
  }
}

/** Redraw every canvas-backed screen plus the HUD. */
function drawScreens() {
  const st = pipe.state;
  const t = pipe.time;
  drawOLED(world.screens.beacon, st, t);
  drawBand(world.screens.band, st, t);
  drawPhone(world.screens.phone, st, t);
  drawLaptop(world.screens.hub, st, t);
  for (const [, dev] of world.devices) {
    if (dev.spectrum && dev.spectrum.sprite.material.opacity > 0.02) {
      drawSpectrum(dev.spectrum.scr, st, t);
    }
  }
  refreshHUD();
}

function renderAll() {
  const lp = document.getElementById('left').getBoundingClientRect();
  const rp = document.getElementById('right').getBoundingClientRect();
  tagLayer.update(innerWidth, innerHeight, {
    l: lp.right + 14, r: rp.left - 14, t: 74, b: innerHeight - 128,
  });

  composer.render();
  if (soloId) {
    renderer.autoClear = false;
    renderer.render(scrimScene, scrimCam);   // dim the world
    renderer.clearDepth();                   // then draw the module on top
    renderer.render(soloScene, camera);
    renderer.autoClear = true;
  }
}

function frame() {
  requestAnimationFrame(frame);
  const raw = Math.min(clock.getDelta(), 0.05);
  advance(raw);
  screenAcc += raw;
  if (screenAcc > 1 / 24) { screenAcc = 0; drawScreens(); }
  renderAll();
}

// ===========================================================================
//  Boot
// ===========================================================================
// Intro figures are derived from config so they can never drift from the BOM.
{
  const cost = (id) => DEVICES.find((d) => d.id === id).bom.reduce((s, [, p]) => s + p, 0);
  const full = DEVICES.reduce((s, d) => s + d.bom.reduce((a, [, p]) => a + p, 0), 0);
  const starter = cost('node-kitchen') + cost('beacon') + cost('shaker');
  const rupees = (n) => '₹' + n.toLocaleString('en-IN');
  document.getElementById('statLatency').textContent =
    STAGES.reduce((s, x) => s + x.ms, 0) + ' ms';
  document.getElementById('statCost').textContent = rupees(full);
  document.getElementById('statCostLabel').textContent =
    `all 6 modules as built here · hostel starter kit from ${rupees(starter)}`;
}

document.getElementById('introGo').addEventListener('click', () => {
  document.getElementById('intro').classList.add('gone');
  controls.flyTo(PRESETS.overview);
});
controls.flyTo({ ...PRESETS.overview, radius: 22, duration: 2.4 });
document.getElementById('loading').classList.add('gone');
frame();

// Console handle — useful for driving the twin from devtools or a demo script.
window.__vas = {
  pipe, world, controls, flags, PRESETS,
  trigger: (id, live = null) => pipe.trigger(id, live),
  ack: () => pipe.ack(),
  /** The live-microphone listener, once it has been started. null otherwise. */
  get listener() { return listener; },
  listen: () => micBtn.click(),
  /** Advance the simulation without waiting on requestAnimationFrame. */
  step(seconds, dt = 1 / 60) {
    const n = Math.max(1, Math.round(seconds / dt));
    for (let i = 0; i < n; i++) advance(dt);
    drawScreens();
    renderAll();
    return pipe.state.mode;
  },
  inspect: (id) => inspect(id),
  close: () => closeInspector(),
};
