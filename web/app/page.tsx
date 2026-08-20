"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ServerEvent, VoiceClient } from "@/lib/voice";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";
// Survives a reload, dies with the tab, and is per-tab - which is what a call
// is. The server holds a dropped call open for a minute, so handing this back
// continues the conversation instead of starting a new one.
const SESSION_KEY = "aria-session";

type Line = { who: "caller" | "agent"; text: string };
type ToolLog = { name: string; args: Record<string, unknown>; ok: boolean; detail: string };

function Meter({ level, label, hue }: { level: number; label: string; hue: string }) {
  const bars = 28;
  return (
    <div className="meter">
      <span className="meter-label">{label}</span>
      <div className="bars">
        {Array.from({ length: bars }, (_, i) => {
          const on = level * bars > i;
          const h = 6 + Math.sin((i / bars) * Math.PI) * 30;
          return (
            <i
              key={i}
              style={{
                height: `${on ? h * (0.5 + level) : 4}px`,
                background: on ? hue : "var(--bar-off)",
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

export default function Home() {
  const clientRef = useRef<VoiceClient | null>(null);
  const [state, setState] = useState<"idle" | "connecting" | "live" | "closed">("idle");
  const [lines, setLines] = useState<Line[]>([]);
  const [partial, setPartial] = useState("");
  const [tools, setTools] = useState<ToolLog[]>([]);
  const [latency, setLatency] = useState<number[]>([]);
  const [meta, setMeta] = useState<{ stt?: string; tts?: string; llm?: string; id?: string }>({});
  const [levels, setLevels] = useState<[number, number]>([0, 0]);
  const [summary, setSummary] = useState<Record<string, unknown> | null>(null);
  const [typed, setTyped] = useState("");
  const logRef = useRef<HTMLDivElement>(null);
  const wanted = useRef<string | null>(null);  // the session id this connect asked for

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [lines, partial]);

  const clearLog = useCallback(() => {
    setLines([]);
    setTools([]);
    setLatency([]);
  }, []);

  const onEvent = useCallback((e: ServerEvent) => {
    switch (e.type) {
      case "ready":
        setMeta({ stt: e.stt, tts: e.tts, llm: e.llm, id: e.session_id });
        // A different id than we asked for means the old call was already
        // closed out, so what is on screen belongs to nothing.
        if (wanted.current && wanted.current !== e.session_id) clearLog();
        sessionStorage.setItem(SESSION_KEY, e.session_id);
        break;
      case "partial":
        setPartial(e.text);
        break;
      case "final":
        setPartial("");
        setLines((l) => [...l, { who: "caller", text: e.text }]);
        break;
      case "assistant":
        setLines((l) => {
          const last = l[l.length - 1];
          if (last?.who === "agent") {
            return [...l.slice(0, -1), { who: "agent", text: `${last.text} ${e.text}` }];
          }
          return [...l, { who: "agent", text: e.text }];
        });
        break;
      case "tool":
        setTools((t) => [
          ...t,
          {
            name: e.name,
            args: e.args,
            ok: Boolean(e.result.ok),
            detail: String(e.result.tier ?? e.result.confirmation ?? e.result.message ?? "done"),
          },
        ]);
        break;
      case "metric":
        setLatency((m) => [...m, e.first_audio_ms]);
        break;
      case "summary":
        setSummary(e.record);
        break;
      case "error":
        setLines((l) => [...l, { who: "agent", text: `[error] ${e.message}` }]);
        break;
    }
  }, [clearLog]);

  async function start() {
    // An id left over from a dropped connection resumes that call; the log on
    // screen is its first half, so it stays.
    const resuming = sessionStorage.getItem(SESSION_KEY);
    wanted.current = resuming;
    if (!resuming) clearLog();
    setSummary(null);
    const client = new VoiceClient(WS_URL, {
      onEvent,
      onLevels: (mic, agent) => setLevels([mic, agent]),
      onState: setState,
    });
    clientRef.current = client;
    try {
      await client.start(resuming ?? undefined);
    } catch (err) {
      setState("idle");
      setLines([{ who: "agent", text: `[mic error] ${(err as Error).message}` }]);
    }
  }

  async function stop() {
    // Hanging up on purpose is the one case that is not resumable, so forget
    // the id before the socket closes - the next call starts clean.
    sessionStorage.removeItem(SESSION_KEY);
    wanted.current = null;
    await clientRef.current?.stop();
    clientRef.current = null;
  }

  const live = state === "live";
  const worst = latency.length ? Math.max(...latency) : 0;
  const avg = latency.length ? Math.round(latency.reduce((a, b) => a + b, 0) / latency.length) : 0;

  return (
    <main>
      <header>
        <div>
          <h1>Aria</h1>
          <p className="sub">Voice support &amp; qualification agent</p>
        </div>
        <div className="stack">
          <span className={`pill ${live ? "on" : ""}`}>{state}</span>
          {meta.llm && <span className="pill muted">{meta.llm}</span>}
          {meta.stt && <span className="pill muted">{meta.stt}</span>}
          {meta.tts && <span className="pill muted">{meta.tts}</span>}
        </div>
      </header>

      <section className="viz">
        <Meter label="caller" level={levels[0]} hue="var(--caller)" />
        <Meter label="agent" level={levels[1]} hue="var(--agent)" />
      </section>

      <section className="controls">
        {live ? (
          <button className="danger" onClick={stop}>
            End call
          </button>
        ) : (
          <button onClick={start} disabled={state === "connecting"}>
            {state === "connecting" ? "Connecting…" : "Start call"}
          </button>
        )}
        <form
          onSubmit={(ev) => {
            ev.preventDefault();
            if (!typed.trim() || !live) return;
            // no optimistic line: the server echoes a typed turn back as
            // `final`, exactly like a spoken one, and two would show up
            clientRef.current?.sendText(typed.trim());
            setTyped("");
          }}
        >
          <input
            value={typed}
            onChange={(ev) => setTyped(ev.target.value)}
            placeholder={live ? "…or type a turn instead" : "Start the call first"}
            disabled={!live}
          />
          {/* a real submit button, not implicit Enter: Enter alone does not
              submit a form without one everywhere, and phones need the tap */}
          <button type="submit" disabled={!live || !typed.trim()}>
            Send
          </button>
        </form>
        {latency.length > 0 && (
          <span className={`pill ${worst <= 1200 ? "on" : "warn"}`}>
            first audio avg {avg} ms · worst {worst} ms
          </span>
        )}
      </section>

      <section className="grid">
        <div className="panel" ref={logRef}>
          <h2>Transcript</h2>
          {lines.length === 0 && !partial && <p className="empty">Nothing yet.</p>}
          {lines.map((l, i) => (
            <p key={i} className={`line ${l.who}`}>
              <b>{l.who}</b>
              {l.text}
            </p>
          ))}
          {partial && (
            <p className="line caller ghost">
              <b>caller</b>
              {partial}
            </p>
          )}
        </div>

        <div className="panel">
          <h2>Tool calls</h2>
          {tools.length === 0 && <p className="empty">No tools called yet.</p>}
          {tools.map((t, i) => (
            <div key={i} className={`tool ${t.ok ? "ok" : "bad"}`}>
              <code>
                {t.name}({Object.values(t.args).join(", ")})
              </code>
              <span>{t.detail}</span>
            </div>
          ))}
          {summary && (
            <>
              <h2 className="spaced">Call record</h2>
              <pre>{JSON.stringify(summary, null, 2)}</pre>
            </>
          )}
        </div>
      </section>
    </main>
  );
}
