"use client";

/**
 * Browser half of the voice loop.
 *
 * Mic -> AudioWorklet -> PCM16 16 kHz -> websocket, and websocket -> PCM16 ->
 * scheduled AudioBuffers. The AudioContext is opened at 16 kHz so the browser
 * does the resampling and the wire format is the PRD's format end to end.
 */

export const SAMPLE_RATE = 16000;

// The server holds a dropped call open for RESUME_GRACE - 60 seconds by
// default - and hands the same session back to whoever asks for its id. So
// there is a minute in which redialling continues the conversation instead of
// starting a new one, and this is the client's half of that minute.
const RECONNECT_WINDOW = 60_000;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export type ServerEvent =
  | { type: "ready"; session_id: string; stt: string; tts: string; llm: string }
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
  onState: (s: "idle" | "connecting" | "live" | "reconnecting" | "closed") => void;
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
  private sessionId?: string;
  private hangingUp = false;

  constructor(private url: string, private h: Handlers) {}

  async start(sessionId?: string) {
    this.hangingUp = false;
    this.sessionId = sessionId;
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

    // Not awaited: the first dial reports itself through onState the way it
    // always did, and a call that never opened has no session to go back for.
    this.connect().catch(() => {});
  }

  /**
   * Open the socket and wire it up. Resolves once it is open, rejects with the
   * close event if it never got there - which is what lets redial tell "still
   * down, try again" apart from "the server does not want this call".
   */
  private connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const qs = new URLSearchParams();
      if (this.sessionId) qs.set("session_id", this.sessionId);
      // Shipped to the browser, so it is a gate against everyone else's page and
      // against scanners, not against this page's own user.
      if (process.env.NEXT_PUBLIC_WS_TOKEN) qs.set("token", process.env.NEXT_PUBLIC_WS_TOKEN);
      const query = qs.toString();
      const ws = new WebSocket(`${this.url}/ws${query ? `?${query}` : ""}`);
      ws.binaryType = "arraybuffer";
      this.ws = ws;
      let opened = false;

      ws.onopen = () => {
        opened = true;
        this.h.onState("live");
        if (this.node) {
          this.node.port.onmessage = (ev) => {
            if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(ev.data as Int16Array);
          };
        }
        this.meter();
        resolve();
      };
      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") {
          const msg = JSON.parse(ev.data) as ServerEvent;
          // Whatever the server calls this call is what a redial asks for. On
          // a first connection that is a new id; after a grace period that
          // expired it is a different one, and asking for the old one again
          // would only get the same answer a second time.
          if (msg.type === "ready") this.sessionId = msg.session_id;
          if (msg.type === "interrupt") this.stopPlayback();
          if (msg.type === "summary") this.summaryArrived?.();
          this.h.onEvent(msg);
        } else {
          this.enqueue(new Int16Array(ev.data as ArrayBuffer));
        }
      };
      ws.onclose = (ev) => {
        if (!opened) return reject(ev);
        // 1008 is the server saying do not come back: a bad token, or a caller
        // who blew the audio rate limit. Anything else is a call that dropped,
        // and redial is where it is decided whether to go back for it.
        if (ev.code === 1008) return this.h.onState("closed");
        void this.redial();
      };
      ws.onerror = () => this.h.onEvent({ type: "error", message: "websocket error" });
    });
  }

  /**
   * Redial the call that just dropped.
   *
   * Only the socket died: the microphone, the AudioContext and the worklet are
   * all still alive, so this reopens the socket and nothing else - the caller
   * is never asked for the microphone twice and the audio graph does not
   * flicker. A dropped connection is not a hangup, and on a phone it is the
   * ordinary case rather than the exception.
   */
  private async redial() {
    const deadline = Date.now() + RECONNECT_WINDOW;
    for (let attempt = 0; Date.now() < deadline; attempt++) {
      // Twice, for two different races: stop() sets the flag before it closes
      // the socket, so the close it causes arrives here with it already set -
      // and a caller who gives up while this is between attempts sets it
      // during the sleep.
      if (this.hangingUp) return;
      this.h.onState("reconnecting");
      await sleep(Math.min(500 * 2 ** attempt, 5000));
      if (this.hangingUp) return;
      try {
        await this.connect();
        return;
      } catch (ev) {
        if ((ev as CloseEvent).code === 1008) break; // the server means it
      }
    }
    this.h.onState("closed");
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
    cancelAnimationFrame(this.raf); // a redial must not leave two loops running
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
    this.hangingUp = true;     // before anything closes: redial checks this
    this.h.onState("closed");  // the wait for the record below is seconds long
    cancelAnimationFrame(this.raf);
    this.stopPlayback();
    this.mic?.getTracks().forEach((t) => t.stop());
    this.node?.disconnect();
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "end" }));
      // wait for the call record, don't guess at how long the summary takes:
      // it is an LLM round trip and runs several seconds behind the hangup.
      // Only worth waiting on a socket that is open - hanging up in the middle
      // of a redial has nothing to wait for, and 15 seconds of it is a hang.
      await new Promise<void>((resolve) => {
        this.summaryArrived = resolve;
        setTimeout(resolve, 15000);
      });
    }
    this.ws?.close();
    await this.ctx?.close();
    this.h.onState("idle");
  }
}
