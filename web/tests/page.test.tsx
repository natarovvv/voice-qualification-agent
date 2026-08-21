/**
 * The page's half of session resume.
 *
 * The server holds a dropped call open for a minute; the only thing that makes
 * that reachable is the browser handing the session id back. That is three
 * lines of sessionStorage, and getting any of them wrong looks exactly like
 * the feature working until someone actually loses their connection.
 */
import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import Home from "@/app/page";

const SESSION_KEY = "aria-session";

const mocked = vi.hoisted(() => ({ clients: [] as MockClient[] }));

type Handlers = {
  onEvent: (e: Record<string, unknown>) => void;
  onLevels: (mic: number, agent: number) => void;
  onState: (s: string) => void;
};

type MockClient = { h: Handlers; started: string | undefined; startCalls: number };

vi.mock("@/lib/voice", () => ({
  SAMPLE_RATE: 16000,
  VoiceClient: class {
    started: string | undefined;
    startCalls = 0;
    constructor(_url: string, readonly h: Handlers) {
      mocked.clients.push(this as unknown as MockClient);
    }
    async start(sessionId?: string) {
      this.started = sessionId;
      this.startCalls += 1;
      this.h.onState("live");
    }
    async stop() {
      this.h.onState("idle");
    }
    sendText() {}
  },
}));

const client = () => mocked.clients[mocked.clients.length - 1];

/** start() is async, and every server event lands as a React state update. */
const settle = (fn: () => void) => act(async () => void fn());

async function ready(session_id: string) {
  await settle(() => fireEvent.click(screen.getByRole("button", { name: /start call/i })));
  await settle(() =>
    client().h.onEvent({ type: "ready", session_id, stt: "x", tts: "y", llm: "z" }),
  );
}

const said = (text: string) =>
  settle(() => client().h.onEvent({ type: "final", speaker: "caller", text }));

/** The connection died - no End call, just gone. */
const dropped = () => settle(() => client().h.onState("closed"));

describe("session resume", () => {
  it("starts a fresh call and remembers the id the server minted", async () => {
    render(<Home />);
    await ready("S1");

    expect(client().started).toBeUndefined();
    expect(sessionStorage.getItem(SESSION_KEY)).toBe("S1");
  });

  it("hands the id back after a drop, and keeps the transcript on screen", async () => {
    render(<Home />);
    await ready("S1");
    await said("what does it cost");
    await dropped();

    await ready("S1");
    expect(client().started).toBe("S1");
    // the call continues, so its first half is still what the caller sees
    expect(screen.getByText(/what does it cost/)).toBeTruthy();
  });

  it("clears the stale transcript when the server hands back a different call", async () => {
    render(<Home />);
    await ready("S1");
    await said("what does it cost");
    await dropped();

    // the grace period expired while we were away: this is a new call
    await ready("S2");
    expect(screen.queryByText(/what does it cost/)).toBeNull();
    expect(sessionStorage.getItem(SESSION_KEY)).toBe("S2");
  });

  it("forgets the id on a deliberate hangup", async () => {
    render(<Home />);
    await ready("S1");

    await settle(() => fireEvent.click(screen.getByRole("button", { name: /end call/i })));
    expect(sessionStorage.getItem(SESSION_KEY)).toBeNull();

    // ...so the next call is a new one, not a resume of a call that ended
    await ready("S2");
    expect(client().started).toBeUndefined();
  });

  it("clears the log when a redial the client made itself lands in a new call", async () => {
    render(<Home />);
    await ready("S1");
    await said("what does it cost");

    // nobody pressed Start: the client redialled, and the grace had expired
    await settle(() => client().h.onState("reconnecting"));
    await settle(() =>
      client().h.onEvent({ type: "ready", session_id: "S2", stt: "x", tts: "y", llm: "z" }),
    );
    expect(screen.queryByText(/what does it cost/)).toBeNull();
    expect(sessionStorage.getItem(SESSION_KEY)).toBe("S2");
  });

  it("offers End call while the client is redialling", async () => {
    render(<Home />);
    await ready("S1");
    await settle(() => client().h.onState("reconnecting"));

    // the call is not over, so Start is not the button that helps - but the
    // caller has to be able to give up on it
    expect(screen.queryByRole("button", { name: /start call/i })).toBeNull();
    expect(screen.getByRole("button", { name: /end call/i })).toBeTruthy();
  });
});
