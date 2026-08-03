// ============================================================================
//  listener.js — mic + front-end + model + the policy that decides to fire.
//
//  The policy is the part that makes this usable rather than merely working.
//  A raw classifier running four times a second on a sliding window will happily
//  emit forty detections for one doorbell, and will fire on any single unlucky
//  frame. Three rules fix that:
//
//    smoothing    a class must clear the gate on SMOOTH_HITS of the last
//                 SMOOTH_WINDOW inferences, so one bad frame cannot fire an alert
//    gate         the same 75% threshold the twin already used
//    refractory   after firing, that class is ignored for REFRACTORY_S, so one
//                 real event produces one alert
//
//  Inference runs in a worker; only this smoothing logic is on the main thread,
//  and it is a few dozen floating point operations per hop.
// ============================================================================
import { DSP } from './dsp-config.js';
import { Mic, micAvailable } from './mic.js';

const WORKER_URL = new URL('./infer-worker.js', import.meta.url);

// Model class -> the twin's event id in js/config.js. `background` maps to nothing.
export const CLASS_TO_EVENT = {
  doorbell: 'doorbell',
  fire_alarm: 'fire',
  baby_cry: 'baby',
  glass_break: 'glass',
  tv_music: 'tv',
  background: null,
};

const ALERT = new Set(['doorbell', 'fire_alarm', 'baby_cry', 'glass_break']);
const FEATURE_LABEL = `log-mel ${DSP.n_bins} × ${DSP.n_frames} → CNN`;
const CLASSES = DSP.classes;

export class Listener {
  /**
   * @param {object} hooks
   *   onDetect(eventId, live)  fire an event in the twin
   *   onProbs(probs, rms, classes)  per-inference telemetry for the HUD
   *   onError(err)
   */
  constructor(hooks = {}) {
    this.hooks = hooks;
    this.history = [];                 // recent probability vectors
    this.lastFired = new Map();        // class -> wall-clock seconds
    this.running = false;
    this.lastRms = 0;
    this.lastSpectrum = null;
    this.worker = null;
    this.inflight = false;             // one window at a time; drop rather than queue
    this.fallback = null;              // main-thread infer(), if workers are unavailable
    this.mic = new Mic({
      onWindow: (w) => this._onWindow(w),
      onLevel: (rms) => { this.lastRms = rms; },
    });
  }

  static available() { return micAvailable(); }

  async start() {
    try {
      await this._startWorker();
      await this.mic.start();
      this.running = true;
      this.history.length = 0;
      this.lastFired.clear();
    } catch (err) {
      this.hooks.onError?.(err);
      throw err;
    }
  }

  async _startWorker() {
    try {
      const w = new Worker(WORKER_URL, { type: 'module' });
      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('inference worker did not start')), 8000);
        w.onmessage = (e) => {
          if (!e.data?.ready) return;
          clearTimeout(timer);
          w.onmessage = (ev) => this._onResult(ev.data);
          resolve();
        };
        w.onerror = (e) => { clearTimeout(timer); reject(e.error || new Error(e.message)); };
      });
      this.worker = w;
    } catch (err) {
      // Module workers are widely supported, but a failure here should degrade to
      // a working (if choppier) twin rather than no microphone at all.
      console.warn('[vas] inference worker unavailable, running on the main thread:', err);
      const mod = await import('./infer.js');
      this.fallback = mod.infer;
    }
  }

  async stop() {
    this.running = false;
    await this.mic.stop();
    if (this.worker) { this.worker.terminate(); this.worker = null; }
    this.fallback = null;
    this.history.length = 0;
    this.inflight = false;
  }

  _onWindow(samples) {
    if (!this.running) return;

    // If the previous window is still being classified, skip this one. Queueing
    // would build unbounded latency and report detections after the sound is gone.
    if (this.inflight) return;

    if (this.worker) {
      this.inflight = true;
      // Copy: mic.js reuses its window buffer for the next hop.
      const copy = samples.slice();
      this.worker.postMessage({ samples: copy }, [copy.buffer]);
    } else if (this.fallback) {
      this._onResult(this.fallback(samples));
    }
  }

  _onResult({ probs, spectrum }) {
    this.inflight = false;
    if (!this.running) return;

    this.lastSpectrum = spectrum;

    this.history.push(probs);
    if (this.history.length > DSP.smooth_window) this.history.shift();

    this.hooks.onProbs?.(probs, this.lastRms, CLASSES);
    this._decide();
  }

  _decide() {
    if (this.history.length < DSP.smooth_window) return;

    // Mean probability across the window, plus how many individual inferences
    // actually cleared the gate. Requiring both stops a single loud frame from
    // firing while still reacting inside about a second.
    const n = CLASSES.length;
    const mean = new Float32Array(n);
    const hits = new Int32Array(n);
    for (const p of this.history) {
      let top = 0;
      for (let i = 0; i < n; i++) {
        mean[i] += p[i] / this.history.length;
        if (p[i] > p[top]) top = i;
      }
      if (p[top] >= DSP.conf_threshold) hits[top]++;
    }

    let best = 0;
    for (let i = 1; i < n; i++) if (mean[i] > mean[best]) best = i;
    const cls = CLASSES[best];

    if (cls === 'background') return;
    if (mean[best] < DSP.conf_threshold || hits[best] < DSP.smooth_hits) return;

    // Refractory is measured against the wall clock, not against a count of
    // processed windows. Under load a window can be dropped (see `inflight`),
    // and a per-window counter would stretch the 8 s window to 16 s of real time
    // without anything looking wrong.
    const now = performance.now() / 1000;
    const last = this.lastFired.get(cls);
    if (last !== undefined && now - last < DSP.refractory_s) return;

    const eventId = CLASS_TO_EVENT[cls];
    if (!eventId) return;
    this.lastFired.set(cls, now);

    // The twin's gate expects "how confident are we that this is an alert".
    // For tv_music that is deliberately the *alert* probability, not the model's
    // own confidence — the model is very sure it is hearing television, and the
    // right consequence of that certainty is to stay silent. Feeding the alert
    // probability through drives the twin's existing suppression path with a real
    // number instead of the 0.31 that used to be hardcoded.
    const confidence = ALERT.has(cls)
      ? mean[best]
      : Math.max(...[...ALERT].map((c) => mean[CLASSES.indexOf(c)]));

    this.hooks.onDetect?.(eventId, {
      confidence,
      className: cls,
      featureLabel: FEATURE_LABEL,
      spectrum: this.lastSpectrum,
      probs: Object.fromEntries(CLASSES.map((c, i) => [c, mean[i]])),
    });
  }

  /** Reset smoothing — call when the twin is reset so a stale window cannot refire. */
  clearHistory() { this.history.length = 0; }
}
