import { roundRect, clamp, noise } from './util.js';
import { CONF_THRESHOLD, ESCALATE_SECONDS } from './config.js';

const MONO = '"SF Mono", "Roboto Mono", Menlo, Consolas, monospace';
const UI = '"Inter", "Helvetica Neue", Arial, sans-serif';

// ===========================================================================
//  SSD1306 OLED on the Alert Beacon — 128×64, monochrome, drawn at 4×
// ===========================================================================
export function drawOLED(scr, st, t) {
  const { ctx, w, h } = scr;
  const S = w / 128;
  ctx.fillStyle = '#04070a';
  ctx.fillRect(0, 0, w, h);

  const ink = '#8ce9ff';
  ctx.textBaseline = 'middle';

  if (st.mode === 'idle') {
    ctx.fillStyle = ink;
    ctx.font = `${7 * S}px ${MONO}`;
    ctx.textAlign = 'center';
    ctx.fillText('SYSTEM ARMED', w / 2, 20 * S);
    ctx.font = `${6 * S}px ${MONO}`;
    ctx.fillStyle = '#3f7f96';
    ctx.fillText(st.wifi ? 'MQTT  OK' : 'ESP-NOW  DIRECT', w / 2, 34 * S);
    ctx.fillText(`${st.nodesOnline}/3 NODES  ${st.battery}%`, w / 2, 46 * S);
    // heartbeat sweep
    ctx.strokeStyle = '#17414f';
    ctx.lineWidth = S;
    ctx.beginPath();
    for (let x = 0; x < 128; x++) {
      const y = 58 + Math.sin((x + t * 30) * 0.25) * 2.5;
      x === 0 ? ctx.moveTo(x * S, y * S) : ctx.lineTo(x * S, y * S);
    }
    ctx.stroke();
    scr.flush();
    return;
  }

  const ev = st.event;
  if (!ev) { scr.flush(); return; }

  // The beacon must not name an event before the node has actually decided.
  if (['sensing', 'classifying', 'transport', 'deciding'].includes(st.mode)) {
    ctx.fillStyle = ink;
    ctx.font = `${7 * S}px ${MONO}`;
    ctx.textAlign = 'center';
    ctx.fillText(st.mode === 'sensing' ? 'SOUND DETECTED' : 'ANALYSING...', w / 2, 20 * S);
    const p = clamp(st.mode === 'sensing' ? st.stageProgress * 0.4 : 0.4 + st.stageProgress * 0.6, 0, 1);
    ctx.strokeStyle = ink;
    ctx.lineWidth = S;
    ctx.strokeRect(14 * S, 32 * S, 100 * S, 10 * S);
    ctx.fillStyle = ink;
    ctx.fillRect(16 * S, 34 * S, 96 * S * p, 6 * S);
    ctx.fillStyle = '#3f7f96';
    ctx.font = `${6 * S}px ${MONO}`;
    ctx.fillText(st.detectorName, w / 2, 52 * S);
    scr.flush();
    return;
  }

  if (st.mode === 'rejected') {
    ctx.fillStyle = '#3f7f96';
    ctx.font = `${7 * S}px ${MONO}`;
    ctx.textAlign = 'center';
    ctx.fillText('BACKGROUND NOISE', w / 2, 24 * S);
    ctx.fillText('IGNORED', w / 2, 40 * S);
    scr.flush();
    return;
  }

  const critical = ev.priority === 'CRITICAL';
  const blink = critical ? Math.floor(t * 4) % 2 === 0 : true;

  if (critical && blink) {
    ctx.fillStyle = ink;
    ctx.fillRect(0, 0, w, 16 * S);
    ctx.fillStyle = '#04070a';
  } else {
    ctx.fillStyle = ink;
  }
  ctx.font = `bold ${9 * S}px ${MONO}`;
  ctx.textAlign = 'center';
  ctx.fillText(ev.oled ? ev.oled[0] : ev.label.toUpperCase(), w / 2, 9 * S);

  ctx.fillStyle = ink;
  ctx.font = `${7 * S}px ${MONO}`;
  ctx.fillText(ev.oled ? ev.oled[1] : ev.where.toUpperCase(), w / 2, 26 * S);

  ctx.fillStyle = '#3f7f96';
  ctx.font = `${6 * S}px ${MONO}`;
  ctx.fillText(`${(st.confidence * 100).toFixed(0)}%  ${st.detectorName}`, w / 2, 38 * S);

  if (st.mode === 'escalated') {
    ctx.fillStyle = ink;
    ctx.font = `bold ${7 * S}px ${MONO}`;
    ctx.fillText('CALLING CONTACT', w / 2, 54 * S);
  } else if (st.mode === 'acked') {
    ctx.fillStyle = ink;
    ctx.font = `${7 * S}px ${MONO}`;
    ctx.fillText('ACKNOWLEDGED', w / 2, 54 * S);
  } else if (ev.escalate) {
    const left = Math.ceil(st.ackLeft);
    ctx.fillStyle = ink;
    ctx.font = `${7 * S}px ${MONO}`;
    ctx.fillText(`ACK IN ${left}s`, w / 2, 52 * S);
    ctx.fillRect(4 * S, 60 * S, 120 * S * (st.ackLeft / ESCALATE_SECONDS), 3 * S);
  } else {
    ctx.fillStyle = '#3f7f96';
    ctx.font = `${6 * S}px ${MONO}`;
    ctx.fillText('PRESS TO CLEAR', w / 2, 54 * S);
  }
  scr.flush();
}

// ===========================================================================
//  Wearable band OLED
// ===========================================================================
export function drawBand(scr, st, t) {
  const { ctx, w, h } = scr;
  ctx.fillStyle = '#04070a';
  ctx.fillRect(0, 0, w, h);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  const live = ['alerting', 'escalated', 'acked'].includes(st.mode) && st.event && !st.event.rejected;
  if (!live) {
    ctx.fillStyle = '#2f6a7a';
    ctx.font = `${w * 0.13}px ${MONO}`;
    ctx.fillText(st.mode === 'idle' ? 'ALL CLEAR' : 'LISTENING…', w / 2, h * 0.4);
    ctx.font = `${w * 0.1}px ${MONO}`;
    ctx.fillText(`${st.battery}%`, w / 2, h * 0.68);
  } else {
    const ev = st.event;
    const blink = ev.priority === 'CRITICAL' ? Math.floor(t * 4) % 2 === 0 : true;
    ctx.fillStyle = blink ? '#8ce9ff' : '#2a5766';
    ctx.font = `bold ${w * 0.15}px ${MONO}`;
    ctx.fillText(ev.label.toUpperCase(), w / 2, h * 0.35);
    ctx.fillStyle = '#4d93a8';
    ctx.font = `${w * 0.11}px ${MONO}`;
    ctx.fillText(ev.where.toUpperCase(), w / 2, h * 0.62);
    ctx.fillText(ev.priority, w / 2, h * 0.85);
  }
  scr.flush();
}

// ===========================================================================
//  Phone app  (also mirrored 1:1 into the HTML side panel)
// ===========================================================================
export function drawPhone(scr, st, t) {
  const { ctx, w, h } = scr;
  const ev = st.event;
  // Only after the decision engine has actually raised the alert — and only if
  // there is a network to push it over.
  const alerting = ['alerting', 'escalated', 'acked'].includes(st.mode)
    && ev && !ev.rejected && st.wifi;

  // background
  if (alerting && ev.priority === 'CRITICAL' && st.mode !== 'acked') {
    const pulse = 0.5 + 0.5 * Math.sin(t * 7);
    ctx.fillStyle = `rgb(${28 + pulse * 40}, ${10 + pulse * 6}, ${14 + pulse * 8})`;
  } else {
    ctx.fillStyle = '#0d1117';
  }
  ctx.fillRect(0, 0, w, h);

  // status bar
  ctx.fillStyle = 'rgba(255,255,255,0.55)';
  ctx.font = `${w * 0.042}px ${UI}`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(st.clock, w * 0.07, h * 0.035);
  ctx.textAlign = 'right';
  ctx.fillText(`${st.wifi ? 'Wi-Fi' : 'offline'}  ▮${st.battery}%`, w * 0.93, h * 0.035);

  // header
  ctx.textAlign = 'left';
  ctx.fillStyle = '#e6edf3';
  ctx.font = `600 ${w * 0.062}px ${UI}`;
  ctx.fillText('SoundSight', w * 0.07, h * 0.085);
  ctx.fillStyle = '#7d8590';
  ctx.font = `${w * 0.038}px ${UI}`;
  ctx.fillText(st.night ? 'Sleep mode · hearing aids out' : 'Normal mode', w * 0.07, h * 0.118);

  const cx = w / 2;

  if (!alerting) {
    // ---- SAFE state ----
    const offlineAlert = !st.wifi && ev && !ev.rejected
      && ['alerting', 'escalated', 'acked'].includes(st.mode);
    const listening = ['sensing', 'classifying', 'transport', 'deciding'].includes(st.mode);
    const r = w * 0.26;
    const cy = h * 0.32;
    if (offlineAlert) {
      // Honest failure mode: no Wi-Fi means no push. The hardware still alerts.
      ctx.strokeStyle = 'rgba(245,158,11,0.9)';
      ctx.lineWidth = w * 0.012;
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = '#f59e0b';
      ctx.font = `600 ${w * 0.075}px ${UI}`;
      ctx.textAlign = 'center';
      ctx.fillText('OFFLINE', cx, cy - w * 0.03);
      ctx.fillStyle = '#adbac7';
      ctx.font = `${w * 0.037}px ${UI}`;
      ctx.fillText('No push — beacon, shaker', cx, cy + w * 0.06);
      ctx.fillText('and band are still alerting', cx, cy + w * 0.115);
      ctx.textAlign = 'left';
    } else {
    const ringCol = listening ? '#38bdf8' : '#22c55e';
    ctx.strokeStyle = listening ? 'rgba(56,189,248,0.22)' : 'rgba(34,197,94,0.22)';
    ctx.lineWidth = w * 0.02;
    ctx.beginPath(); ctx.arc(cx, cy, r + Math.sin(t * (listening ? 6 : 2)) * w * 0.012, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = ringCol;
    ctx.lineWidth = w * 0.012;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = listening ? '#38bdf8' : '#22c55e';
    ctx.font = `600 ${listening ? w * 0.072 : w * 0.11}px ${UI}`;
    ctx.textAlign = 'center';
    ctx.fillText(listening ? 'ANALYSING' : 'SAFE', cx, cy - w * 0.02);
    ctx.fillStyle = '#7d8590';
    ctx.font = `${w * 0.04}px ${UI}`;
    ctx.fillText(listening ? st.detectorName : `${st.nodesOnline} modules listening`, cx, cy + w * 0.09);
    ctx.textAlign = 'left';
    }
  } else {
    // ---- ALERT state ----
    const col = ev.css;
    const y0 = h * 0.16;
    ctx.fillStyle = col;
    roundRect(ctx, w * 0.06, y0, w * 0.88, h * 0.245, w * 0.045);
    ctx.globalAlpha = st.mode === 'acked' ? 0.14 : 0.2;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = col;
    ctx.lineWidth = w * 0.006;
    ctx.stroke();

    ctx.textAlign = 'left';
    ctx.fillStyle = col;
    ctx.font = `700 ${w * 0.04}px ${UI}`;
    ctx.fillText(`${ev.priority} PRIORITY`, w * 0.11, y0 + h * 0.032);
    ctx.fillStyle = '#e6edf3';
    ctx.font = `700 ${w * 0.068}px ${UI}`;
    ctx.fillText(ev.phone.title, w * 0.11, y0 + h * 0.083);
    ctx.fillStyle = '#adbac7';
    ctx.font = `${w * 0.042}px ${UI}`;
    ctx.fillText(ev.where, w * 0.11, y0 + h * 0.13);
    ctx.fillStyle = '#7d8590';
    ctx.font = `${w * 0.036}px ${MONO}`;
    ctx.fillText(`${(st.confidence * 100).toFixed(0)}% · ${st.detectorName}`, w * 0.11, y0 + h * 0.172);
    ctx.fillText(`${st.totalMs} ms end-to-end`, w * 0.11, y0 + h * 0.205);

    // ---- action button ----
    const by = h * 0.45;
    const bh = h * 0.085;
    if (st.mode === 'acked') {
      ctx.fillStyle = 'rgba(34,197,94,0.15)';
      roundRect(ctx, w * 0.06, by, w * 0.88, bh, w * 0.03); ctx.fill();
      ctx.fillStyle = '#22c55e';
      ctx.font = `600 ${w * 0.05}px ${UI}`;
      ctx.textAlign = 'center';
      ctx.fillText('✓  ACKNOWLEDGED', cx, by + bh / 2);
    } else if (st.mode === 'escalated') {
      const f = Math.floor(t * 2) % 2 === 0;
      ctx.fillStyle = f ? '#ef4444' : 'rgba(239,68,68,0.35)';
      roundRect(ctx, w * 0.06, by, w * 0.88, bh, w * 0.03); ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.font = `600 ${w * 0.046}px ${UI}`;
      ctx.textAlign = 'center';
      ctx.fillText('CALLING EMERGENCY CONTACT', cx, by + bh / 2);
    } else {
      ctx.fillStyle = '#e6edf3';
      roundRect(ctx, w * 0.06, by, w * 0.88, bh, w * 0.03); ctx.fill();
      ctx.fillStyle = '#0d1117';
      ctx.font = `700 ${w * 0.052}px ${UI}`;
      ctx.textAlign = 'center';
      ctx.fillText('ACKNOWLEDGE', cx, by + bh / 2);
      if (ev.escalate) {
        ctx.fillStyle = '#ef4444';
        ctx.font = `${w * 0.038}px ${MONO}`;
        ctx.fillText(`escalating in ${Math.ceil(st.ackLeft)}s`, cx, by + bh + h * 0.028);
      }
    }
  }

  // ---- history ----
  const hy = h * 0.60;
  ctx.textAlign = 'left';
  ctx.fillStyle = '#7d8590';
  ctx.font = `600 ${w * 0.038}px ${UI}`;
  ctx.fillText('RECENT', w * 0.07, hy);
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(w * 0.07, hy + h * 0.018); ctx.lineTo(w * 0.93, hy + h * 0.018); ctx.stroke();

  const rows = st.history.slice(0, 5);
  if (!rows.length) {
    ctx.fillStyle = '#4b5563';
    ctx.font = `${w * 0.04}px ${UI}`;
    ctx.fillText('No events yet', w * 0.07, hy + h * 0.055);
  }
  rows.forEach((r, i) => {
    const y = hy + h * 0.055 + i * h * 0.055;
    ctx.fillStyle = r.css;
    ctx.beginPath(); ctx.arc(w * 0.085, y, w * 0.016, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#adbac7';
    ctx.font = `${w * 0.042}px ${UI}`;
    ctx.fillText(r.label, w * 0.13, y);
    ctx.fillStyle = '#4b5563';
    ctx.font = `${w * 0.034}px ${MONO}`;
    ctx.textAlign = 'right';
    ctx.fillText(r.time, w * 0.93, y);
    ctx.textAlign = 'left';
  });

  // ---- tab bar ----
  ctx.fillStyle = 'rgba(255,255,255,0.04)';
  ctx.fillRect(0, h * 0.93, w, h * 0.07);
  const tabs = ['Home', 'History', 'Settings'];
  tabs.forEach((tb, i) => {
    ctx.fillStyle = i === 0 ? '#e6edf3' : '#4b5563';
    ctx.font = `${w * 0.038}px ${UI}`;
    ctx.textAlign = 'center';
    ctx.fillText(tb, w * (0.2 + i * 0.3), h * 0.963);
  });

  scr.flush();
}

// ===========================================================================
//  Hub laptop — MQTT broker + decision engine console
// ===========================================================================
export function drawLaptop(scr, st, t) {
  const { ctx, w, h } = scr;
  ctx.fillStyle = '#080b10';
  ctx.fillRect(0, 0, w, h);

  // title bar
  ctx.fillStyle = '#11161f';
  ctx.fillRect(0, 0, w, h * 0.085);
  ctx.fillStyle = '#7d8590';
  ctx.font = `${h * 0.042}px ${MONO}`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText('mosquitto  ·  decision_engine.py', w * 0.03, h * 0.043);
  ctx.textAlign = 'right';
  ctx.fillStyle = st.wifi ? '#22c55e' : '#f59e0b';
  ctx.fillText(st.wifi ? '● broker up' : '● broker unreachable — nodes on ESP-NOW', w * 0.97, h * 0.043);

  ctx.textAlign = 'left';
  ctx.font = `${h * 0.04}px ${MONO}`;
  const lines = st.console.slice(-11);
  lines.forEach((l, i) => {
    ctx.fillStyle = l.color || '#5b6673';
    ctx.fillText(l.text, w * 0.03, h * 0.145 + i * h * 0.072);
  });

  // caret
  if (Math.floor(t * 2) % 2 === 0) {
    ctx.fillStyle = '#22c55e';
    ctx.fillText('▌', w * 0.03, h * 0.145 + lines.length * h * 0.072);
  }
  scr.flush();
}

// ===========================================================================
//  Live MFCC / spectrum strip drawn above the listening node
// ===========================================================================
export function drawSpectrum(scr, st, t) {
  const { ctx, w, h } = scr;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = 'rgba(6,10,16,0.92)';
  roundRect(ctx, 0, 0, w, h, h * 0.12);
  ctx.fill();
  ctx.strokeStyle = st.specColor + '55';
  ctx.lineWidth = 2;
  ctx.stroke();

  const bars = st.spectrum || [];
  const n = bars.length || 1;
  const pad = w * 0.05;
  const bw = (w - pad * 2) / n;
  const base = h * 0.86;
  for (let i = 0; i < n; i++) {
    const jitter = 0.86 + 0.14 * noise(i * 3.7 + Math.floor(t * 22) * 0.11);
    const v = clamp(bars[i] * st.specGain * jitter, 0, 1);
    const bh = v * (h * 0.66);
    ctx.fillStyle = st.specColor;
    ctx.globalAlpha = 0.35 + v * 0.65;
    ctx.fillRect(pad + i * bw + bw * 0.15, base - bh, bw * 0.7, bh);
  }
  ctx.globalAlpha = 1;

  ctx.font = `${h * 0.115}px ${MONO}`;
  ctx.textBaseline = 'alphabetic';
  // Right-hand readout wins the space; the left label is clipped if they collide.
  ctx.textAlign = 'right';
  ctx.fillStyle = st.specColor;
  ctx.fillText(st.specRight, w - pad, h * 0.985);
  const rightW = ctx.measureText(st.specRight).width;
  ctx.textAlign = 'left';
  ctx.fillStyle = '#8b98a8';
  const maxW = w - pad * 2 - rightW - h * 0.12;
  let label = st.specLabel;
  while (label.length > 4 && ctx.measureText(label).width > maxW) label = label.slice(0, -1);
  ctx.fillText(label, pad, h * 0.985);
  scr.flush();
}

export const confPasses = (c) => c >= CONF_THRESHOLD;
