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
