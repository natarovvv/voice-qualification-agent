/**
 * Stand-ins for the browser APIs the voice client talks to.
 *
 * These are deliberately dumb recorders rather than simulators: the point is
 * to see exactly what the client asked the browser to do - when each buffer
 * was scheduled, what went out on the socket - because that is the logic worth
 * testing. Anything they pretend to do beyond recording is a chance for the
 * fake to agree with a bug.
 */
import { vi } from "vitest";

export class FakeAudioBuffer {
  readonly duration: number;
  private readonly data: Float32Array;

  constructor(readonly length: number, readonly sampleRate: number) {
    this.data = new Float32Array(length);
    this.duration = length / sampleRate;
  }

  getChannelData(): Float32Array {
    return this.data;
  }
}

export class FakeBufferSource {
  buffer: FakeAudioBuffer | null = null;
  startedAt: number | null = null;
  stopped = false;
  onended: (() => void) | null = null;

  connect() {}

  start(when: number) {
    this.startedAt = when;
  }

  stop() {
    this.stopped = true;
  }
}

class FakeAnalyser {
  fftSize = 512;
  frequencyBinCount = 256;
  connect() {}
  getByteTimeDomainData(into: Uint8Array) {
    into.fill(128); // 128 is silence in the byte time domain
  }
}

export class FakeAudioContext {
  readonly sampleRate: number;
  currentTime = 0;
  closed = false;
  readonly destination = {};
  readonly audioWorklet = { addModule: vi.fn(async () => {}) };
  /** Every source the client created, in the order it created them. */
  readonly created: FakeBufferSource[] = [];

  constructor(opts: { sampleRate: number }) {
    this.sampleRate = opts.sampleRate;
  }

  createBuffer(_channels: number, length: number, rate: number) {
    return new FakeAudioBuffer(length, rate);
  }

  createBufferSource() {
    const src = new FakeBufferSource();
    this.created.push(src);
    return src;
  }

  createGain() {
    return { connect: () => {}, gain: { value: 1 } };
  }

  createAnalyser() {
    return new FakeAnalyser();
  }

  createMediaStreamSource() {
    return { connect: () => {} };
  }

  async close() {
    this.closed = true;
  }
}

export class FakeWorkletNode {
  port: { onmessage: ((ev: { data: unknown }) => void) | null } = { onmessage: null };
  disconnected = false;
  disconnect() {
    this.disconnected = true;
  }
  /** Hand the client a frame of microphone audio, as the real worklet does. */
  emit(frame: Int16Array) {
    this.port.onmessage?.({ data: frame });
  }
}

export class FakeWebSocket {
  static readonly OPEN = 1;
  static readonly CLOSED = 3;
  /** The socket the client opened most recently. */
  static last: FakeWebSocket | null = null;

  readyState = FakeWebSocket.OPEN;
  binaryType = "";
  readonly sent: unknown[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string | ArrayBuffer }) => void) | null = null;
  onclose: ((ev: { code: number }) => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(readonly url: string) {
    FakeWebSocket.last = this;
  }

  send(data: unknown) {
    this.sent.push(data);
  }

  close(code = 1000) {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code });
  }

  // --- the server's half, driven by the test ---

  open() {
    this.onopen?.();
  }

  json(msg: unknown) {
    this.onmessage?.({ data: JSON.stringify(msg) });
  }

  audio(pcm: Int16Array) {
    this.onmessage?.({ data: pcm.buffer as ArrayBuffer });
  }

  /** The socket dying under the call, which is not the same as hanging up.
   *  1006 is what a browser reports when the connection just went away. */
  drop(code = 1006) {
    this.close(code);
  }

  /** What the client sent, with the JSON control frames parsed. */
  get control(): Record<string, unknown>[] {
    return this.sent
      .filter((s): s is string => typeof s === "string")
      .map((s) => JSON.parse(s));
  }
}

export type Installed = {
  ctx: FakeAudioContext;
  node: FakeWorkletNode;
  micDenied: boolean;
};

/** Put the fakes in place of the browser APIs. Undone by vi.unstubAllGlobals. */
export function installBrowser({ denyMic = false } = {}): Installed {
  const state: Installed = {
    ctx: null as unknown as FakeAudioContext,
    node: null as unknown as FakeWorkletNode,
    micDenied: denyMic,
  };

  vi.stubGlobal(
    "AudioContext",
    class extends FakeAudioContext {
      constructor(opts: { sampleRate: number }) {
        super(opts);
        state.ctx = this;
      }
    },
  );
  vi.stubGlobal(
    "AudioWorkletNode",
    class extends FakeWorkletNode {
      constructor() {
        super();
        state.node = this;
      }
    },
  );
  vi.stubGlobal("WebSocket", FakeWebSocket);
  FakeWebSocket.last = null;

  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: {
      getUserMedia: vi.fn(async () => {
        if (denyMic) throw new Error("Permission denied");
        return { getTracks: () => [{ stop: vi.fn() }] };
      }),
    },
  });

  return state;
}

/** PCM16 of the given length, so a duration is easy to reason about. */
export function pcm(samples: number, value = 1000): Int16Array {
  return new Int16Array(samples).fill(value);
}
