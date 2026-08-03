import * as THREE from 'three';
import {
  EVENT_BY_ID, DEVICE_BY_ID, STAGES, OFFLINE_TRANSPORT,
  ESCALATE_SECONDS, ESCALATE_SIM, CONF_THRESHOLD,
} from './config.js';
import { clamp, lerp, smoothstep, noise, makeHalo, V } from './util.js';

const STAGE_INDEX = Object.fromEntries(STAGES.map((s, i) => [s.id, i]));

/**
 * Bed-shaker policy.
 *
 * Asleep with hearing aids out, the shaker is the ONLY channel that reaches the
 * user — so anything that actually matters must fire it. But waking someone at
 * 03:00 for a doorbell is its own harm, so LOW priority is held back overnight.
 * Awake, the beacon and band are enough for everything short of a fire.
 */
export function shakerFires(priority, night) {
  if (priority === 'CRITICAL') return true;
  if (night) return priority === 'HIGH' || priority === 'MEDIUM';
  return false;
}

// ===========================================================================
//  Expanding acoustic wavefront
// ===========================================================================
class SoundWave {
  constructor(scene, origin, color, maxR = 9, life = 1.6) {
    this.scene = scene;
    this.life = life;
    this.t = 0;
    this.maxR = maxR;
    this.group = new THREE.Group();
    this.group.position.copy(origin);
    this.rings = [];
    for (let i = 0; i < 3; i++) {
      const m = new THREE.Mesh(
        new THREE.RingGeometry(0.94, 1.0, 72),
        new THREE.MeshBasicMaterial({
          color, transparent: true, opacity: 0, side: THREE.DoubleSide,
          depthWrite: false, blending: THREE.AdditiveBlending,
        }),
      );
      m.rotation.x = -Math.PI / 2;
      m.position.y = -origin.y + 0.05;   // rings hug the floor
      this.group.add(m);
      this.rings.push({ mesh: m, delay: i * 0.22 });
    }
    this.shell = new THREE.Mesh(
      new THREE.SphereGeometry(1, 24, 16),
      new THREE.MeshBasicMaterial({
        color, transparent: true, opacity: 0, wireframe: true,
        depthWrite: false, blending: THREE.AdditiveBlending,
      }),
    );
    this.group.add(this.shell);
    scene.add(this.group);
  }

  update(dt) {
    this.t += dt;
    const k = this.t / this.life;
    for (const r of this.rings) {
      const rk = clamp((this.t - r.delay) / (this.life - r.delay), 0, 1);
      const rad = 0.15 + rk * this.maxR;
      r.mesh.scale.setScalar(rad);
      r.mesh.material.opacity = rk <= 0 ? 0 : (1 - rk) * 0.55;
    }
    const sk = clamp(k * 1.15, 0, 1);
    this.shell.scale.setScalar(0.15 + sk * this.maxR * 0.72);
    this.shell.material.opacity = (1 - sk) * 0.18;
    return this.t < this.life;
  }

  dispose() {
    this.scene.remove(this.group);
    this.group.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
  }
}

// ===========================================================================
//  Message packet flying along a link
// ===========================================================================
class Packet {
  constructor(scene, from, to, color, duration = 0.75, label = '') {
    this.scene = scene;
    this.duration = duration;
    this.t = 0;
    this.label = label;
    const mid = from.clone().add(to).multiplyScalar(0.5);
    mid.y = Math.max(from.y, to.y) + from.distanceTo(to) * 0.22 + 0.35;
    this.curve = new THREE.QuadraticBezierCurve3(from.clone(), mid, to.clone());

    const pts = this.curve.getPoints(48);
    this.line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({
        color, transparent: true, opacity: 0.28, depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    scene.add(this.line);

    this.dot = new THREE.Mesh(
      new THREE.SphereGeometry(0.045, 12, 10),
      new THREE.MeshBasicMaterial({ color, toneMapped: false }),
    );
    scene.add(this.dot);
    this.halo = makeHalo(color, 0.5);
    this.halo.material.opacity = 1;
    scene.add(this.halo);
    this.pos = from.clone();
  }

  update(dt) {
    this.t += dt;
    const k = clamp(this.t / this.duration, 0, 1);
    this.curve.getPoint(k, this.pos);
    this.dot.position.copy(this.pos);
    this.halo.position.copy(this.pos);
    const fade = k > 0.85 ? 1 - (k - 0.85) / 0.15 : 1;
    this.line.material.opacity = 0.3 * fade;
    this.halo.material.opacity = fade;
    this.dot.material.opacity = fade;
    return this.t < this.duration;
  }

  dispose() {
    this.scene.remove(this.line);
    this.scene.remove(this.dot);
    this.scene.remove(this.halo);
    this.line.geometry.dispose();
    this.line.material.dispose();
    this.dot.geometry.dispose();
    this.dot.material.dispose();
  }
}

// ===========================================================================
//  Pipeline
// ===========================================================================
export class Pipeline {
  constructor(scene, world, hooks = {}) {
    this.scene = scene;
    this.world = world;
    this.hooks = hooks;
    this.effects = [];
    this.time = 0;
    this.clockSeconds = 22 * 3600 + 47 * 60 + 12;

    this.state = {
      mode: 'idle',
      event: null,
      detector: null,
      detectorName: '—',
      detectors: [],
      confidence: 0,
      stage: -1,
      stageProgress: 0,
      ackLeft: ESCALATE_SECONDS,
      totalMs: 0,
      history: [],
      console: [
        { text: '$ mosquitto -c /etc/mosquitto.conf', color: '#5b6673' },
        { text: '  listener 1883 · persistence on', color: '#3f4954' },
        { text: '$ python3 decision_engine.py', color: '#5b6673' },
        { text: '  subscribed  home/+/event', color: '#3f4954' },
        { text: '  3 nodes online · awaiting events', color: '#22c55e' },
      ],
      wifi: true,
      night: true,
      battery: 94,
      mains: true,
      nodesOnline: 3,
      clock: '22:47',
      spectrum: null,
      specColor: '#38bdf8',
      specGain: 0,
      specLabel: '',
      specRight: '',
    };

    this.seq = null;      // active stage sequence
    this.clearAt = null;  // auto-clear timer for non-critical events
  }

  // -------------------------------------------------------------------
  log(text, color) {
    this.state.console.push({ text, color });
    if (this.state.console.length > 40) this.state.console.shift();
    this.hooks.onLog?.(text, color);
  }

  timeString() {
    const total = this.clockSeconds % 86400;
    const s = Math.floor(total);
    const ms = Math.floor((total - s) * 1000);
    const hh = String(Math.floor(s / 3600)).padStart(2, '0');
    const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
    const ss = String(s % 60).padStart(2, '0');
    return {
      hm: `${hh}:${mm}`,
      hms: `${hh}:${mm}:${ss}`,
      full: `${hh}:${mm}:${ss}.${String(ms).padStart(3, '0')}`,
    };
  }

  /**
   * The animation is deliberately slower than reality so it can be watched.
   * The virtual clock therefore runs at whatever rate makes the log honest:
   * during the pipeline it advances by each stage's real millisecond cost, and
   * during the acknowledgement window it advances at the true 30 s rate.
   */
  clockRate() {
    const st = this.state;
    if (this.seq) {
      const cur = STAGES[st.stage];
      if (cur) {
        const ms = (cur.id === 'transport' && !st.wifi) ? OFFLINE_TRANSPORT.ms : cur.ms;
        return (ms / 1000) / cur.sim;
      }
    }
    if (st.mode === 'alerting' && st.event?.escalate) return ESCALATE_SECONDS / ESCALATE_SIM;
    return 1;
  }

  // -------------------------------------------------------------------
  get busy() {
    return this.state.mode !== 'idle';
  }

  /** Confidence for the running event: the model's real output when the twin is
   *  driven by the microphone, the scripted figure from config.js otherwise. */
  get conf() {
    const st = this.state;
    if (st.live) return st.live.confidence;
    return st.event ? st.event.confidence : 0;
  }

  /**
   * @param {string} eventId
   * @param {?{confidence:number, spectrum:number[], className:string, probs:object}} live
   *   Present when the twin is being driven by the microphone rather than by a
   *   scripted trigger. Carries the model's actual confidence and its actual
   *   log-mel frame, so the console, the gate and the spectrum display all show
   *   what was really heard instead of the figures baked into config.js.
   */
  trigger(eventId, live = null) {
    const ev = EVENT_BY_ID[eventId];
    if (!ev) return;
    if (this.busy) this.reset(true);

    const src = new THREE.Vector3(...ev.source);

    // Which modules can hear it, nearest first.
    const heard = [];
    for (const [id, dev] of this.world.devices) {
      const def = dev.def;
      if (!def.range || !def.classes.includes(ev.id)) continue;
      const p = new THREE.Vector3(...def.pos);
      const d = p.distanceTo(src);
      if (d <= def.range) heard.push({ id, name: def.name, dist: d, room: def.room });
    }
    heard.sort((a, b) => a.dist - b.dist);

    if (!heard.length) {
      this.log(`! ${ev.label} outside every module's range`, '#f59e0b');
      return;
    }

    const st = this.state;
    st.event = ev;
    st.detectors = heard;
    st.detector = heard[0].id;
    st.detectorName = heard[0].name;
    st.confidence = 0;
    st.live = live;
    st.mode = 'sensing';
    st.stage = 0;
    st.stageProgress = 0;
    st.ackLeft = ESCALATE_SECONDS;
    st.spectrum = live?.spectrum ?? ev.mfcc;
    st.specColor = ev.css;
    st.specGain = 0;
    st.totalMs = 0;
    this.clearAt = null;

    this.seq = { i: 0, t: 0 };
    this.effects.push(new SoundWave(this.scene, src, ev.color, 11, 1.7));
    this.sourcePos = src;

    const tm = this.timeString();
    this.log(`[${tm.full}] acoustic event @ ${ev.where}`, '#5b6673');
    this.hooks.onTrigger?.(ev, heard);
  }

  ack() {
    const st = this.state;
    if (st.mode !== 'alerting' && st.mode !== 'escalated') return;
    const wasEscalated = st.mode === 'escalated';
    st.mode = 'acked';
    const tm = this.timeString();
    this.log(`[${tm.full}] ACK received from phone${wasEscalated ? ' — call cancelled' : ''}`, '#22c55e');
    const h = st.history[0];
    if (h) h.acked = true;
    this.clearAt = 2.2;
    this.hooks.onAck?.();
  }

  reset(silent = false) {
    const st = this.state;
    st.mode = 'idle';
    st.event = null;
    st.live = null;
    st.stage = -1;
    st.stageProgress = 0;
    st.confidence = 0;
    st.specGain = 0;
    st.detectors = [];
    this.seq = null;
    this.clearAt = null;
    for (const e of this.effects) e.dispose();
    this.effects.length = 0;
    if (!silent) this.hooks.onIdle?.();
  }

  // -------------------------------------------------------------------
  devicePos(id, localOffset = null) {
    const dev = this.world.devices.get(id);
    if (!dev) return V(0, 0, 0);
    const p = new THREE.Vector3();
    dev.group.getWorldPosition(p);
    if (localOffset) p.add(localOffset);
    return p;
  }

  stageDef(i) { return STAGES[i]; }

  advanceStage() {
    const st = this.state;
    const ev = st.event;
    const prev = STAGES[st.stage];
    st.stage++;
    st.stageProgress = 0;
    const cur = STAGES[st.stage];
    const tm = this.timeString();

    // ---- account real-world latency as each stage completes ----
    if (prev) {
      if (prev.id === 'transport') {
        st.totalMs += st.wifi ? prev.ms : OFFLINE_TRANSPORT.ms;
      } else {
        st.totalMs += prev.ms;
      }
    }

    if (!cur) {                       // pipeline finished → steady alert
      st.mode = ev.escalate ? 'alerting' : 'alerting';
      if (!ev.escalate) this.clearAt = 9;
      return;
    }

    switch (cur.id) {
      case 'capture': {
        st.mode = 'sensing';
        st.specLabel = 'I²S 16 kHz · 1024-sample window';
        st.specRight = 'CAPTURING';
        this.log(`[${tm.full}] ${st.detectorName}: capturing 1024-sample window…`, '#5b6673');
        break;
      }
      case 'classify': {
        st.mode = 'classifying';
        st.specLabel = st.live?.featureLabel ?? 'log-mel 40 × 49 → CNN';
        this.log(`[${tm.full}] ${st.detectorName}: buffer full (64 ms) → inference`, '#5b6673');
        break;
      }
      case 'transport': {
        if (this.conf < CONF_THRESHOLD) {
          // ---- rejection path ----
          st.mode = 'rejected';
          st.specRight = 'REJECTED';
          st.specLabel = st.live ? `heard ${st.live.className}` : 'unknown / background';
          this.log(`[${tm.full}] ${st.detectorName}: unknown ${(this.conf * 100).toFixed(0)}% < ${CONF_THRESHOLD * 100}% → suppressed`, '#f59e0b');
          st.history.unshift({
            time: tm.hm, label: `${ev.label} — suppressed`, where: ev.where,
            priority: 'IGNORE', css: '#64748b', acked: true, suppressed: true,
          });
          st.history = st.history.slice(0, 12);
          this.hooks.onHistory?.();
          this.clearAt = 2.6;
          this.seq = null;
          return;
        }
        st.mode = 'transport';
        const from = this.devicePos(st.detector);
        if (st.wifi) {
          this.effects.push(new Packet(this.scene, from, this.devicePos('hub', V(0, 0.15, 0)), ev.color, cur.sim));
          this.log(`[${tm.full}] MQTT ▸ home/${this.world.devices.get(st.detector).def.room}/event  {"c":"${ev.id}","p":${this.conf.toFixed(2)}}`, ev.css);
        } else {
          this.effects.push(new Packet(this.scene, from, this.devicePos('beacon', V(0, 0.25, 0)), ev.color, cur.sim));
          this.log(`[${tm.full}] ESP-NOW ▸ beacon direct (broker unreachable)`, '#f59e0b');
        }
        // secondary detectors report too — the hub will de-duplicate
        for (const d of st.detectors.slice(1)) {
          const p2 = this.devicePos(d.id);
          const target = st.wifi ? this.devicePos('hub', V(0, 0.15, 0)) : this.devicePos('beacon', V(0, 0.25, 0));
          this.effects.push(new Packet(this.scene, p2, target, ev.color, cur.sim * 1.15));
        }
        break;
      }
      case 'decide': {
        st.mode = 'deciding';
        const dup = st.detectors.length > 1
          ? ` (${st.detectors.length} nodes, de-duplicated)` : '';
        this.log(`[${tm.full}] ${st.wifi ? 'hub' : 'beacon'}: ${ev.label} → priority ${ev.priority}${dup}`, ev.css);
        break;
      }
      case 'alert': {
        st.mode = 'alerting';
        const from = st.wifi ? this.devicePos('hub', V(0, 0.15, 0)) : this.devicePos('beacon', V(0, 0.25, 0));
        if (st.wifi) {
          this.effects.push(new Packet(this.scene, from, this.devicePos('beacon', V(0, 0.25, 0)), ev.color, cur.sim));
        }
        this.effects.push(new Packet(this.scene, this.devicePos('beacon', V(0, 0.25, 0)), this.devicePos('shaker', V(0, 0.08, 0)), ev.color, cur.sim * 1.1));
        this.effects.push(new Packet(this.scene, this.devicePos('beacon', V(0, 0.25, 0)), this.devicePos('band', V(0, 0.08, 0)), ev.color, cur.sim * 1.2));
        this.log(`[${tm.full}] beacon: ${ev.ledPattern} · ${ev.vibration}`, ev.css);
        if (st.night && !shakerFires(ev.priority, true)) {
          this.log(`[${tm.full}] sleep mode: ${ev.priority} priority — bed shaker held, beacon only`, '#8b98a8');
        }

        const tm2 = this.timeString();
        st.history.unshift({
          time: tm2.hm, label: ev.label, where: ev.where,
          priority: ev.priority, css: ev.css, acked: false,
        });
        st.history = st.history.slice(0, 12);
        this.hooks.onHistory?.();
        break;
      }
      case 'notify': {
        if (st.wifi) {
          this.effects.push(new Packet(this.scene, this.devicePos('hub', V(0, 0.15, 0)), this.devicePos('phone', V(0, 0.1, 0)), ev.color, cur.sim));
          this.log(`[${tm.full}] push ▸ phone: "${ev.phone.title}"`, '#5b6673');
        } else {
          this.log(`[${tm.full}] phone unreachable — hardware alerts unaffected`, '#f59e0b');
        }
        break;
      }
    }
  }

  // -------------------------------------------------------------------
  update(dt) {
    this.time += dt;
    this.clockSeconds += dt * this.clockRate();
    const st = this.state;
    st.clock = this.timeString().hm;

    // effects
    for (let i = this.effects.length - 1; i >= 0; i--) {
      if (!this.effects[i].update(dt)) {
        this.effects[i].dispose();
        this.effects.splice(i, 1);
      }
    }

    // stage sequencing
    if (this.seq) {
      const cur = STAGES[st.stage];
      if (cur) {
        this.seq.t += dt;
        st.stageProgress = clamp(this.seq.t / cur.sim, 0, 1);

        if (cur.id === 'capture') st.specGain = smoothstep(st.stageProgress * 1.6);
        if (cur.id === 'classify') {
          st.confidence = this.conf * smoothstep(st.stageProgress);
          st.specRight = `${(st.confidence * 100).toFixed(0)}%  ${st.event.id}`;
          st.specGain = 1;
        }
        if (st.stageProgress >= 1) {
          this.seq.t = 0;
          this.advanceStage();
        }
      } else {
        this.seq = null;
      }
    }

    // Acknowledgement countdown — starts only once the alert is fully raised,
    // so the 30 s window is measured from the alert, not from mid-pipeline.
    if (!this.seq && st.mode === 'alerting' && st.event?.escalate) {
      st.ackLeft -= dt * (ESCALATE_SECONDS / ESCALATE_SIM);
      if (st.ackLeft <= 0) {
        st.ackLeft = 0;
        st.mode = 'escalated';
        const tm = this.timeString();
        this.log(`[${tm.full}] NO ACK IN ${ESCALATE_SECONDS}s → escalating`, '#ef4444');
        this.log(`[${tm.full}] calling  Warden, Block C  +91 ●●●●● ●●210`, '#ef4444');
        this.hooks.onEscalate?.();
      }
    }

    // auto clear
    if (this.clearAt !== null) {
      this.clearAt -= dt;
      if (this.clearAt <= 0) {
        this.clearAt = null;
        this.reset();
      }
    }

    this.applyVisuals(dt);
  }

  // -------------------------------------------------------------------
  //  Drive every light, LED, screen and shake from the current state
  // -------------------------------------------------------------------
  applyVisuals(dt) {
    const st = this.state;
    const t = this.time;
    const ev = st.event;
    const w = this.world;

    // ---- ambient / night lighting ----
    const wantAmb = st.night ? 8.0 : 16.0;
    for (const l of w.roomLights) l.intensity = lerp(l.intensity, wantAmb, dt * 3);

    // ---- node LEDs ----
    for (const [id, dev] of w.devices) {
      if (!dev.led || !dev.def.range) continue;
      const isDetector = st.detector === id;
      const isSecondary = st.detectors.some((d) => d.id === id) && !isDetector;
      let color = 0x22c55e, inten = 0.35, haloOp = 0.12, haloScale = 1;

      if (st.mode === 'idle') {
        // slow "armed" breathing
        inten = 0.3 + 0.12 * Math.sin(t * 1.6 + id.length);
        haloOp = 0.1 + 0.05 * Math.sin(t * 1.6 + id.length);
      } else if (isDetector || isSecondary) {
        const mag = isDetector ? 1 : 0.55;
        if (st.mode === 'sensing') {
          color = 0xffffff;
          inten = (0.5 + 0.5 * Math.abs(Math.sin(t * 9))) * mag;
          haloOp = 0.55 * mag;
        } else if (st.mode === 'classifying') {
          color = 0x38bdf8;
          inten = (0.55 + 0.45 * Math.abs(Math.sin(t * 14))) * mag;
          haloOp = 0.7 * mag;
        } else if (st.mode === 'rejected') {
          color = 0x64748b;
          inten = 0.5 * mag;
          haloOp = 0.3 * mag;
        } else if (ev) {
          color = ev.color;
          inten = (0.6 + 0.4 * Math.sin(t * 8)) * mag;
          haloOp = 0.8 * mag;
          haloScale = 1.25;
        }
      }

      dev.led.material.emissive.setHex(color);
      dev.led.material.emissiveIntensity = inten * 2.2;
      if (dev.halo) {
        dev.halo.material.color.setHex(color);
        dev.halo.material.opacity = lerp(dev.halo.material.opacity, haloOp, dt * 12);
        dev.halo.scale.setScalar(0.42 * haloScale);
      }

      // spectrum billboard follows the detector only
      if (dev.spectrum) {
        const show = (isDetector || isSecondary) &&
          ['sensing', 'classifying', 'rejected'].includes(st.mode) && st.stage >= 1;
        const target = show ? (isDetector ? 1 : 0.45) : 0;
        dev.spectrum.sprite.material.opacity = lerp(dev.spectrum.sprite.material.opacity, target, dt * 8);
      }
    }

    // ---- hub ----
    const hub = w.devices.get('hub');
    if (hub?.halo) {
      let op = st.wifi ? 0.18 : 0.05;
      if (st.mode === 'deciding') op = 0.7 + 0.3 * Math.sin(t * 18);
      hub.halo.material.color.setHex(st.wifi ? 0x22c55e : 0xf59e0b);
      hub.halo.material.opacity = lerp(hub.halo.material.opacity, op, dt * 10);
    }

    // ---- beacon ring, room flood, shaker, band ----
    const beacon = w.devices.get('beacon');
    const alerting = ['alerting', 'escalated'].includes(st.mode) && ev && !ev.rejected;
    const acked = st.mode === 'acked' && ev;

    let ringI = 0, ringColor = 0x2c3a4d;
    if (alerting) {
      ringColor = ev.color;
      ringI = this.ledPattern(ev.id, t);
    } else if (acked) {
      ringColor = 0x22c55e;
      ringI = 0.5;
    }

    if (beacon?.led) {
      beacon.led.material.emissive.setHex(ringColor);
      beacon.led.material.emissiveIntensity = 0.35 + ringI * 5.5;
      beacon.halo.material.color.setHex(ringColor);
      beacon.halo.material.opacity = lerp(beacon.halo.material.opacity, 0.1 + ringI * 0.9, dt * 20);
      beacon.halo.scale.setScalar(0.5 + ringI * 0.55);
      // vibration shake
      const vib = alerting ? (ev.priority === 'CRITICAL' ? 1 : ringI) : 0;
      beacon.group.position.x = beacon.def.pos[0] + (vib ? (noise(t * 90) - 0.5) * 0.008 : 0);
      beacon.group.position.z = beacon.def.pos[2] + (vib ? (noise(t * 90 + 7) - 0.5) * 0.008 : 0);
    }

    // room flood — the source room and the room the user is in
    const floodRooms = new Set();
    if (alerting && ev.flood) {
      floodRooms.add('bedroom');
      const det = st.detector && DEVICE_BY_ID[st.detector];
      if (det) floodRooms.add(det.room);
    }
    for (const [rid, f] of Object.entries(w.floods)) {
      const on = floodRooms.has(rid);
      const strobe = on ? (Math.sin(t * 25) > -0.2 ? 1 : 0.12) : 0;
      f.plane.material.color.setHex(ev ? ev.color : 0xff2222);
      f.plane.material.opacity = lerp(f.plane.material.opacity, strobe * 0.14, dt * 22);
      f.light.color.setHex(ev ? ev.color : 0xff2222);
      f.light.intensity = lerp(f.light.intensity, strobe * 13, dt * 22);
    }

    // ceiling strip in the bedroom mirrors the beacon
    if (w.ceilingStrip) {
      const on = alerting ? ringI : (acked ? 0.35 : 0);
      w.ceilingStrip.mesh.material.emissive.setHex(alerting || acked ? ringColor : 0x223044);
      w.ceilingStrip.mesh.material.emissiveIntensity = 0.25 + on * 1.9;
      w.ceilingStrip.halo.material.color.setHex(ringColor);
      w.ceilingStrip.halo.material.opacity = lerp(w.ceilingStrip.halo.material.opacity, on * 0.3, dt * 18);
    }

    // bed shaker + the sleeping user reacting
    const shaker = w.devices.get('shaker');
    const shakeOn = alerting && shakerFires(ev.priority, st.night);
    if (shaker) {
      const amp = shakeOn ? (ev.priority === 'CRITICAL' ? 0.012 : 0.006 * this.ledPattern(ev.id, t)) : 0;
      shaker.group.position.x = shaker.def.pos[0] + (noise(t * 120) - 0.5) * amp;
      shaker.group.position.z = shaker.def.pos[2] + (noise(t * 120 + 3) - 0.5) * amp;
      shaker.halo.material.color.setHex(ringColor);
      shaker.halo.material.opacity = lerp(shaker.halo.material.opacity, shakeOn ? 0.45 : 0, dt * 15);
    }
    if (w.person) {
      const stir = shakeOn ? 1 : 0;
      w.person.position.y = lerp(w.person.position.y, stir ? 0.02 : 0, dt * 6)
        + (stir ? (noise(t * 40) - 0.5) * 0.006 : 0);
    }

    const band = w.devices.get('band');
    if (band) {
      const amp = alerting ? 0.005 * (ev.priority === 'CRITICAL' ? 1 : this.ledPattern(ev.id, t)) : 0;
      band.group.position.x = band.def.pos[0] + (noise(t * 110 + 1) - 0.5) * amp;
      band.group.position.z = band.def.pos[2] + (noise(t * 110 + 5) - 0.5) * amp;
      band.halo.material.color.setHex(ringColor);
      band.halo.material.opacity = lerp(band.halo.material.opacity, alerting ? 0.35 : 0, dt * 15);
    }

    const phone = w.devices.get('phone');
    if (phone?.halo) {
      const on = alerting && st.wifi ? 0.4 : 0;
      phone.halo.material.color.setHex(ringColor);
      phone.halo.material.opacity = lerp(phone.halo.material.opacity, on, dt * 12);
    }

    // ---- source props ----
    if (w.doorbellBtn) {
      const on = ev?.id === 'doorbell' && st.stage <= 1 ? 1 : 0.25;
      w.doorbellBtn.halo.material.opacity = lerp(w.doorbellBtn.halo.material.opacity, on * 0.6, dt * 10);
      w.doorbellBtn.led.material.emissiveIntensity = 0.6 + on * 2;
    }
    if (w.smoke) {
      const firing = ev?.id === 'fire' && st.mode !== 'idle';
      const b = firing ? (Math.sin(t * 16) > 0 ? 1 : 0) : 0.15;
      w.smoke.led.material.emissiveIntensity = 0.3 + b * 3;
      w.smoke.halo.material.opacity = lerp(w.smoke.halo.material.opacity, b * 0.8, dt * 20);
    }
    if (w.cooker) {
      const on = ev?.id === 'cooker' && st.mode !== 'idle';
      w.cooker.whistle.rotation.y += on ? dt * 14 : 0;
      w.cooker.group.position.y = 0.975 + (on ? (noise(t * 60) - 0.5) * 0.004 : 0);
    }
    if (w.tvScreen) {
      const on = ev?.id === 'tv' && st.mode !== 'idle';
      w.tvScreen.material.emissiveIntensity = 0.5 + (on ? 0.5 + 0.4 * Math.sin(t * 12) : 0.1 * Math.sin(t * 3));
    }
    if (w.visitor) {
      const near = ev?.id === 'doorbell' && st.mode !== 'idle';
      w.visitor.position.x = lerp(w.visitor.position.x, near ? -7.75 : -8.6, dt * 2.2);
    }
  }

  /** Per-event LED rhythm, 0..1. */
  ledPattern(id, t) {
    switch (id) {
      case 'fire':     return Math.sin(t * 25) > -0.15 ? 1 : 0.05;
      case 'doorbell': { const p = (t % 2.0); return p < 0.28 || (p > 0.5 && p < 0.78) ? 1 : 0.06; }
      case 'cooker':   { const p = (t % 2.2); return (p < 0.2 || (p > 0.4 && p < 0.6) || (p > 0.8 && p < 1.0)) ? 1 : 0.06; }
      case 'baby':     return 0.25 + 0.75 * (0.5 + 0.5 * Math.sin(t * 2.4));
      case 'glass':    { const p = (t % 1.4); return (p < 0.12 || (p > 0.24 && p < 0.36)) ? 1 : 0.06; }
      default:         return 0.5;
    }
  }
}
