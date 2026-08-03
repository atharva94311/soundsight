// ============================================================================
//  capture-worklet.js — runs on the audio render thread.
//
//  Its only job is to batch the 128-sample blocks the browser hands us into
//  hop-sized chunks and post them to the main thread. Doing the batching here
//  rather than on the main thread means 4 messages a second instead of 125.
//
//  Loaded with ctx.audioWorklet.addModule(); config arrives via processorOptions
//  rather than an import, because the worklet global scope is not the page's
//  module scope.
// ============================================================================
class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { hopSamples } = options.processorOptions;
    this.hop = hopSamples;
    this.buf = new Float32Array(this.hop);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;                 // mic not yet delivering; keep the node alive

    for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === this.hop) {
        // Transfer the buffer rather than copy it, then start a fresh one.
        this.port.postMessage(this.buf.buffer, [this.buf.buffer]);
        this.buf = new Float32Array(this.hop);
        this.n = 0;
      }
    }
    return true;
  }
}

registerProcessor('vas-capture', CaptureProcessor);
