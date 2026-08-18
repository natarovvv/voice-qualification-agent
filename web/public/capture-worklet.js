// Mic capture: float32 -> PCM16, batched to 512 samples (32 ms at 16 kHz)
// so the frame size matches the server's VAD window.
// The AudioContext is created at 16 kHz, so the browser has already
// resampled for us - no resampling code lives here.
const FRAME = 512;

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Int16Array(FRAME);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      const s = Math.max(-1, Math.min(1, ch[i]));
      this.buf[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.n === FRAME) {
        const out = this.buf.slice();
        this.port.postMessage(out, [out.buffer]);
        this.n = 0;
      }
    }
    return true;
  }
}

registerProcessor("capture", CaptureProcessor);
