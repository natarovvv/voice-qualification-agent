"use client";

/**
 * Browser half of the voice loop.
 *
 * Mic -> AudioWorklet -> PCM16 16 kHz -> websocket, and websocket -> PCM16 ->
 * scheduled AudioBuffers. The AudioContext is opened at 16 kHz so the browser
 * does the resampling and the wire format is the PRD's format end to end.
 */

export const SAMPLE_RATE = 16000;

export type ServerEvent =
  | { type: "ready"; session_id: string; stt: string; llm: string }
  | { type: "partial"; text: string }
  | { type: "final"; speaker: string; text: string }
  | { type: "assistant"; text: string }
  | { type: "tool"; name: string; args: Record<string, unknown>; result: Record<string, unknown> }
  | { type: "interrupt" }
  | { type: "metric"; first_audio_ms: number }
  | { type: "summary"; record: Record<string, unknown> }
  | { type: "error"; message: string };

type Handlers = {
  onEvent: (e: ServerEvent) => void;
  onLevels: (mic: number, agent: number) => void;
  onState: (s: "idle" | "connecting" | "live" | "closed") => void;
};

function rms(data: Uint8Array): number {
  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    const v = (data[i] - 128) / 128;
    sum += v * v;
  }
  return Math.min(1, Math.sqrt(sum / data.length) * 3);
}

export class VoiceClient {
  private ws?: WebSocket;
  private ctx?: AudioContext;
  private mic?: MediaStream;
  private node?: AudioWorkletNode;
  private micAnalyser?: AnalyserNode;
  private outAnalyser?: AnalyserNode;
  private outGain?: GainNode;
  private queued: AudioBufferSourceNode[] = [];
  private playAt = 0;
  private raf = 0;
  private summaryArrived?: () => void;

  constructor(private url: string, private h: Handlers) {}

  async start(sessionId?: string) {
    this.h.onState("connecting");
    this.ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    await this.ctx.audioWorklet.addModule("/capture-worklet.js");

    // No mic is not a dead call: typed turns run the same server pipeline, and
    // the agent's audio still plays. Denying the mic must not cost you both.
    try {
      this.mic = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      const src = this.ctx.createMediaStreamSource(this.mic);
      this.micAnalyser = this.ctx.createAnalyser();
      this.micAnalyser.fftSize = 512;
      this.node = new AudioWorkletNode(this.ctx, "capture");
      src.connect(this.micAnalyser);
      src.connect(this.node);
    } catch (err) {
      this.h.onEvent({
        type: "error",
        message: `no microphone (${(err as Error).message}) - typed turns only`,
      });
    }

    this.outGain = this.ctx.createGain();
    this.outAnalyser = this.ctx.createAnalyser();
    this.outAnalyser.fftSize = 512;
    this.outGain.connect(this.outAnalyser);
    this.outGain.connect(this.ctx.destination);

    const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
    this.ws = new WebSocket(`${this.url}/ws${qs}`);
    this.ws.binaryType = "arraybuffer";

    this.ws.onopen = () => {
      this.h.onState("live");
      if (this.node) {
        this.node.port.onmessage = (ev) => {
          if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(ev.data as Int16Array);
        };
      }
      this.meter();
    };
    this.ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        const msg = JSON.parse(ev.data) as ServerEvent;
        if (msg.type === "interrupt") this.stopPlayback();
        if (msg.type === "summary") this.summaryArrived?.();
        this.h.onEvent(msg);
      } else {
        this.enqueue(new Int16Array(ev.data as ArrayBuffer));
      }
    };
    this.ws.onclose = () => this.h.onState("closed");
    this.ws.onerror = () => this.h.onEvent({ type: "error", message: "websocket error" });
  }

  /** Type a turn instead of speaking it - same server pipeline. */
  sendText(text: string) {
    this.ws?.send(JSON.stringify({ type: "text", text }));
  }

  private enqueue(pcm: Int16Array) {
    if (!this.ctx || !this.outGain) return;
    const buf = this.ctx.createBuffer(1, pcm.length, SAMPLE_RATE);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;

    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.outGain);
    // 60 ms of slack absorbs network jitter without being audible as lag
    const now = this.ctx.currentTime;
    this.playAt = Math.max(this.playAt, now + 0.06);
    src.start(this.playAt);
    this.playAt += buf.duration;
    this.queued.push(src);
    src.onended = () => {
      this.queued = this.queued.filter((s) => s !== src);
    };
  }

  /** Barge-in: drop everything already scheduled. */
  private stopPlayback() {
    for (const s of this.queued) {
      try {
        s.stop();
      } catch {
        /* already finished */
      }
    }
    this.queued = [];
    this.playAt = this.ctx?.currentTime ?? 0;
  }

  private meter() {
    const micData = new Uint8Array(this.micAnalyser?.frequencyBinCount ?? 0);
    const outData = new Uint8Array(this.outAnalyser!.frequencyBinCount);
    const tick = () => {
      this.micAnalyser?.getByteTimeDomainData(micData);
      this.outAnalyser!.getByteTimeDomainData(outData);
      this.h.onLevels(this.micAnalyser ? rms(micData) : 0, rms(outData));
      this.raf = requestAnimationFrame(tick);
    };
    tick();
  }

  async stop() {
    this.h.onState("closed");  // the wait for the record below is seconds long
    cancelAnimationFrame(this.raf);
    this.stopPlayback();
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: "end" }));
    this.mic?.getTracks().forEach((t) => t.stop());
    this.node?.disconnect();
    // wait for the call record, don't guess at how long the summary takes:
    // it is an LLM round trip and runs several seconds behind the hangup
    await new Promise<void>((resolve) => {
      this.summaryArrived = resolve;
      setTimeout(resolve, 15000);
    });
    this.ws?.close();
    await this.ctx?.close();
    this.h.onState("idle");
  }
}
