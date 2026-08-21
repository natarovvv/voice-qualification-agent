/**
 * The browser half of the voice loop.
 *
 * Everything here is about what the client tells the browser to do: when each
 * chunk of the agent's voice is scheduled to play, what happens to the queue
 * when the caller barges in, and what goes out on the socket. Those are the
 * parts with real failure modes - overlapping audio, dead air after an
 * interrupt, a microphone frame sent into a closed socket.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SAMPLE_RATE, ServerEvent, VoiceClient } from "@/lib/voice";

import { FakeWebSocket, Installed, installBrowser, pcm } from "./fakes";

const LEAD = 0.06; // the jitter buffer's slack, in seconds
const URL_BASE = "ws://localhost:8000";

let browser: Installed;
let events: ServerEvent[];
let states: string[];

function makeClient() {
  events = [];
  states = [];
  return new VoiceClient(URL_BASE, {
    onEvent: (e) => events.push(e),
    onLevels: () => {},
    onState: (s) => states.push(s),
  });
}

/** A started client with its socket open, which is where a call really begins. */
async function connected(sessionId?: string) {
  const client = makeClient();
  await client.start(sessionId);
  const ws = FakeWebSocket.last!;
  ws.open();
  return { client, ws };
}

beforeEach(() => {
  browser = installBrowser();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("playing the agent's voice", () => {
  it("schedules chunks back to back, behind one jitter buffer", async () => {
    const { ws } = await connected();
    ws.audio(pcm(160)); // 10 ms at 16 kHz
    ws.audio(pcm(160));

    const [first, second] = browser.ctx.created;
    expect(first.startedAt).toBeCloseTo(LEAD, 6);
    // exactly where the first one ends: a gap is a stutter, an overlap is a
    // garble, and both are audible
    expect(second.startedAt).toBeCloseTo(LEAD + 0.01, 6);
  });

  it("does not schedule into the past when the queue has already drained", async () => {
    const { ws } = await connected();
    ws.audio(pcm(160));

    browser.ctx.currentTime = 5; // the agent paused; the queue emptied long ago
    ws.audio(pcm(160));

    // a chunk scheduled at 0.07 while the clock reads 5 plays instantly, which
    // is the jitter buffer quietly disappearing
    expect(browser.ctx.created[1].startedAt).toBeCloseTo(5 + LEAD, 6);
  });

  it("converts PCM16 to float without clipping the negative rail", async () => {
    const { ws } = await connected();
    ws.audio(Int16Array.from([0, 32767, -32768, 16384]));

    const samples = browser.ctx.created[0].buffer!.getChannelData();
    expect(samples[0]).toBe(0);
    expect(samples[1]).toBeCloseTo(32767 / 32768, 6);
    // dividing by 32767 instead would put this at -1.000031, which clips
    expect(samples[2]).toBe(-1);
    expect(samples[3]).toBe(0.5);
  });

  it("opens the context at the sample rate the wire format uses", async () => {
    await connected();
    expect(browser.ctx.sampleRate).toBe(SAMPLE_RATE);
  });
});

describe("barge-in", () => {
  it("drops every scheduled chunk and lets the next sentence start now", async () => {
    const { ws } = await connected();
    ws.audio(pcm(16000)); // a full second of the agent talking
    ws.audio(pcm(16000));

    browser.ctx.currentTime = 0.5; // the caller cuts in half a second in
    ws.json({ type: "interrupt" });

    expect(browser.ctx.created.every((s) => s.stopped)).toBe(true);

    ws.audio(pcm(160));
    // without the playAt rewind the reply would queue behind two abandoned
    // seconds of audio: the caller hears nothing for two seconds
    expect(browser.ctx.created[2].startedAt).toBeCloseTo(0.5 + LEAD, 6);
  });

  it("still hands the interrupt to the UI", async () => {
    const { ws } = await connected();
    ws.json({ type: "interrupt" });
    expect(events).toContainEqual({ type: "interrupt" });
  });
});

describe("the socket", () => {
  it("asks to resume the session it is given, and not otherwise", async () => {
    await connected("abc-123");
    expect(FakeWebSocket.last!.url).toBe(`${URL_BASE}/ws?session_id=abc-123`);

    await connected();
    expect(FakeWebSocket.last!.url).toBe(`${URL_BASE}/ws`);
  });

  it("carries the shared token when one is built in", async () => {
    vi.stubEnv("NEXT_PUBLIC_WS_TOKEN", "s3cret");
    await connected("abc-123");
    const url = new URL(FakeWebSocket.last!.url.replace("ws://", "http://"));
    expect(url.searchParams.get("token")).toBe("s3cret");
    expect(url.searchParams.get("session_id")).toBe("abc-123");
  });

  it("sends microphone frames only while it is open", async () => {
    const { ws } = await connected();
    browser.node.emit(pcm(512));
    expect(ws.sent).toHaveLength(1);

    ws.close();
    browser.node.emit(pcm(512));
    expect(ws.sent).toHaveLength(1); // a send on a closed socket throws
  });

  it("sends a typed turn as a control frame", async () => {
    const { client, ws } = await connected();
    client.sendText("what does it cost");
    expect(ws.control).toContainEqual({ type: "text", text: "what does it cost" });
  });
});

describe("a caller with no microphone", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    browser = installBrowser({ denyMic: true });
  });

  it("still gets a call: the agent speaks and typed turns work", async () => {
    const { client, ws } = await connected();

    expect(events).toContainEqual(
      expect.objectContaining({ type: "error", message: expect.stringContaining("no microphone") }),
    );
    expect(states).toContain("live");

    ws.audio(pcm(160));
    expect(browser.ctx.created).toHaveLength(1); // the agent is audible

    client.sendText("hello");
    expect(ws.control).toContainEqual({ type: "text", text: "hello" });
  });
});

describe("hanging up", () => {
  it("waits for the call record instead of guessing at how long it takes", async () => {
    const { client, ws } = await connected();
    let done = false;
    const stopping = client.stop().then(() => {
      done = true;
    });

    expect(ws.control).toContainEqual({ type: "end" });
    await Promise.resolve();
    expect(done).toBe(false); // the summary is an LLM round trip behind

    ws.json({ type: "summary", record: { session_id: "abc" } });
    await stopping;
    expect(done).toBe(true);
    expect(browser.ctx.closed).toBe(true);
  });
});

describe("a call whose socket dropped", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  /** A live call, named by the server, whose socket then went away. */
  async function dropped(code = 1006) {
    const { client } = await connected();
    const first = FakeWebSocket.last!;
    first.json({ type: "ready", session_id: "call-7", stt: "x", tts: "y", llm: "z" });
    first.drop(code);
    return { client, first };
  }

  it("redials the same call instead of ending it", async () => {
    const { first } = await dropped();
    expect(states).toContain("reconnecting");

    await vi.advanceTimersByTimeAsync(500);
    const second = FakeWebSocket.last!;
    expect(second).not.toBe(first);
    // the id the server minted, not the one Start asked for - Start asked for
    // nothing, and the whole point is to land back in the same conversation
    expect(second.url).toContain("session_id=call-7");

    second.open();
    expect(states.at(-1)).toBe("live");
  });

  it("does not ask for the microphone a second time", async () => {
    await dropped();
    await vi.advanceTimersByTimeAsync(500);
    FakeWebSocket.last!.open();
    // only the socket died; a permission prompt in the middle of a call would
    // be worse than the drop it is trying to recover from
    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1);
  });

  it("stays down when the server says not to come back", async () => {
    const { first } = await dropped(1008);
    await vi.advanceTimersByTimeAsync(30_000);
    expect(FakeWebSocket.last).toBe(first);
    expect(states.at(-1)).toBe("closed");
  });

  it("does not redial a call the caller hung up", async () => {
    const { client, ws } = await connected();
    const stopping = client.stop();
    ws.json({ type: "summary", record: {} });
    await stopping;

    await vi.advanceTimersByTimeAsync(30_000);
    expect(FakeWebSocket.last).toBe(ws);
    expect(states.at(-1)).toBe("idle");
    // not even briefly: hanging up is not a drop, and a call that is over must
    // not flash "reconnecting" on its way out
    expect(states).not.toContain("reconnecting");
  });

  it("stops redialling when the caller gives up between attempts", async () => {
    const { client, first } = await dropped();
    expect(states.at(-1)).toBe("reconnecting");

    await client.stop(); // mid-sleep, with nothing open to say goodbye on
    await vi.advanceTimersByTimeAsync(30_000);
    expect(FakeWebSocket.last).toBe(first);
  });

  it("backs off instead of hammering a server that is down", async () => {
    const { first } = await dropped();
    let last: FakeWebSocket = first;
    const at: number[] = [];
    const started = Date.now();

    while (at.length < 3) {
      await vi.advanceTimersByTimeAsync(100);
      if (FakeWebSocket.last !== last) {
        last = FakeWebSocket.last!;
        at.push(Date.now() - started);
        last.close(1006); // still gone
      }
    }

    // half a second, then one, then two. A server that just restarted is the
    // likeliest reason every one of its calls dropped at once, and a tab
    // retrying twice a second for a minute is how it stays down.
    expect(at).toEqual([500, 1500, 3500]);
  });

  it("keeps trying often enough that a late recovery is still caught", async () => {
    const { first } = await dropped();
    let last: FakeWebSocket = first;
    const gone = () => {
      if (FakeWebSocket.last !== last) {
        last = FakeWebSocket.last!;
        last.close(1006);
      }
    };

    const started = Date.now();
    while (Date.now() - started < 35_000) {
      await vi.advanceTimersByTimeAsync(1000);
      gone();
    }

    // the network comes back late in the window. Doubling for ever would have
    // the next attempt half a minute out by now - past the point where the
    // server still has the call - so the backoff has a ceiling.
    for (let i = 0; i < 10 && FakeWebSocket.last === last; i++) {
      await vi.advanceTimersByTimeAsync(1000);
    }
    expect(FakeWebSocket.last).not.toBe(last);

    FakeWebSocket.last!.open();
    expect(states.at(-1)).toBe("live");
  });

  it("gives up once the server would have written the record anyway", async () => {
    const { first } = await dropped();
    const started = Date.now();
    let last: FakeWebSocket = first;

    // every redial finds the server still gone
    for (let i = 0; i < 30 && states.at(-1) !== "closed"; i++) {
      await vi.advanceTimersByTimeAsync(5000);
      if (FakeWebSocket.last !== last) {
        last = FakeWebSocket.last!;
        last.close(1006); // never opened
      }
    }

    expect(states.at(-1)).toBe("closed");
    // and it kept trying for the whole minute the call was resumable
    expect(Date.now() - started).toBeGreaterThanOrEqual(60_000);
  });

  it("does not wait for a record that is not coming when the caller gives up mid-redial", async () => {
    const { client } = await dropped();
    // no open socket, so no "end" to send and no summary to wait for: the
    // fifteen second wait for one would be a fifteen second hang
    await client.stop();
    expect(states.at(-1)).toBe("idle");
  });

  it("leaves one meter loop running, not two", async () => {
    const pending: Array<() => void> = [];
    vi.stubGlobal("requestAnimationFrame", (cb: () => void) => pending.push(cb));
    vi.stubGlobal("cancelAnimationFrame", (h: number) => {
      pending[h - 1] = () => {};
    });
    const frame = () => pending.splice(0).forEach((cb) => cb());

    let ticks = 0;
    const client = new VoiceClient(URL_BASE, {
      onEvent: () => {},
      onLevels: () => {
        ticks += 1;
      },
      onState: () => {},
    });
    await client.start();
    FakeWebSocket.last!.open();
    FakeWebSocket.last!.drop();
    await vi.advanceTimersByTimeAsync(500);
    FakeWebSocket.last!.open();

    ticks = 0;
    frame();
    // two loops would burn twice the frames for one call, quietly, forever
    expect(ticks).toBe(1);
  });
});
