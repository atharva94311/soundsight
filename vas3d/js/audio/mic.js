// ============================================================================
//  mic.js — microphone capture and the sliding inference window.
//
//  Maintains a 1-second ring buffer fed by the capture worklet, and fires a
//  callback every hop (250 ms) with the most recent second of audio. That
//  overlap matters: a doorbell lasting 300 ms would fall across the boundary of
//  non-overlapping windows about a third of the time and be half-heard twice
//  instead of heard once.
// ============================================================================
import { DSP } from './dsp-config.js';

const WORKLET_URL = new URL('./capture-worklet.js', import.meta.url);

export class Mic {
  constructor({ onWindow, onLevel } = {}) {
    this.onWindow = onWindow;       // (Float32Array(n_samples)) => void
    this.onLevel = onLevel;         // (rms: number) => void — for the HUD meter
    this.ctx = null;
    this.stream = null;
    this.node = null;
    this.running = false;

    this.ring = new Float32Array(DSP.n_samples);
    this.write = 0;
    this.filled = 0;
    this.window = new Float32Array(DSP.n_samples);
    this.actualSampleRate = null;
  }

  /**
   * Requests the mic and starts the worklet. Must be called from a user gesture —
   * browsers will not grant getUserMedia or resume an AudioContext otherwise.
   */
  async start() {
    if (this.running) return;

    // Every one of these constraints is deliberately off. The browser's voice
    // processing is built for speech on calls: AGC would flatten the level
    // differences that separate a doorbell from a fire alarm, noise suppression
    // would treat a smoke alarm as stationary noise and remove it, and echo
    // cancellation would gate audio while the room is otherwise quiet. The model
    // was trained on raw audio, so it has to receive raw audio.
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    // Asking for the model's rate lets the browser do the resampling in native
    // code. If it refuses, we resample ourselves rather than feed the model audio
    // at the wrong rate — which would shift every formant and quietly wreck it.
    this.ctx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: DSP.sample_rate,
      latencyHint: 'interactive',
    });
    this.actualSampleRate = this.ctx.sampleRate;
    this.needsResample = Math.abs(this.actualSampleRate - DSP.sample_rate) > 1;
    if (this.needsResample) {
      console.warn(
        `[vas] AudioContext gave ${this.actualSampleRate} Hz, not ${DSP.sample_rate} Hz; ` +
        'resampling in JS. Detection still works but is marginally less accurate.');
    }

    await this.ctx.audioWorklet.addModule(WORKLET_URL);
    if (this.ctx.state === 'suspended') await this.ctx.resume();

    const src = this.ctx.createMediaStreamSource(this.stream);
    const hopSamples = Math.round(DSP.infer_hop_s * this.actualSampleRate);
    this.node = new AudioWorkletNode(this.ctx, 'vas-capture', {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      channelCount: 1,
      processorOptions: { hopSamples },
    });
    this.node.port.onmessage = (e) => this._onChunk(new Float32Array(e.data));
    src.connect(this.node);

    this.running = true;
  }

  async stop() {
    this.running = false;
    if (this.node) { this.node.port.onmessage = null; this.node.disconnect(); this.node = null; }
    if (this.stream) { this.stream.getTracks().forEach((t) => t.stop()); this.stream = null; }
    if (this.ctx) { await this.ctx.close(); this.ctx = null; }
    this.ring.fill(0);
    this.write = 0;
    this.filled = 0;
  }

  /** Linear resampling — only used when the browser refused our sample rate. */
  _resample(chunk) {
    const ratio = DSP.sample_rate / this.actualSampleRate;
    const out = new Float32Array(Math.round(chunk.length * ratio));
    for (let i = 0; i < out.length; i++) {
      const pos = i / ratio;
      const i0 = Math.floor(pos);
      const frac = pos - i0;
      const a = chunk[Math.min(i0, chunk.length - 1)];
      const b = chunk[Math.min(i0 + 1, chunk.length - 1)];
      out[i] = a + (b - a) * frac;
    }
    return out;
  }

  _onChunk(chunk) {
    if (!this.running) return;
    if (this.needsResample) chunk = this._resample(chunk);

    let rms = 0;
    for (let i = 0; i < chunk.length; i++) {
      const s = chunk[i];
      rms += s * s;
      this.ring[this.write] = s;
      this.write = (this.write + 1) % this.ring.length;
    }
    this.filled = Math.min(this.filled + chunk.length, this.ring.length);
    if (this.onLevel) this.onLevel(Math.sqrt(rms / chunk.length));

    // Wait for a full second before classifying; a partly-zero window looks like
    // an event with a sharp onset.
    if (this.filled < this.ring.length || !this.onWindow) return;

    // Unwrap the ring, oldest sample first.
    const n = this.ring.length;
    const head = n - this.write;
    this.window.set(this.ring.subarray(this.write), 0);
    this.window.set(this.ring.subarray(0, this.write), head);
    this.onWindow(this.window);
  }
}

/** True if this browser/context can capture at all. getUserMedia needs a secure
 *  context, which is why the twin must be served over http://localhost, not file://. */
export function micAvailable() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.AudioWorkletNode);
}
