# Voice AI Support & Qualification Agent

Real-time voice-to-voice agent for inbound support and lead qualification.
Streaming audio both ways over one websocket, barge-in, three callable tools,
session memory, and a structured call record at hangup.

Runs on free tiers only. With **no API keys at all** it still runs end to end:
free Microsoft edge-tts for the voice, local whisper for transcription, and a
scripted responder in place of the LLM.

```
browser mic ─ AudioWorklet ─► PCM16 16k ─ websocket ─► VAD ─► STT
                                                              │
        speaker ◄─ scheduled AudioBuffers ◄─ PCM16 ◄─ TTS ◄─ LLM + tools
```

## Layout

| Path | What it is |
|---|---|
| [server/main.py](server/main.py) | FastAPI websocket, turn taking, barge-in, metrics |
| [server/vad.py](server/vad.py) | Silero VAD, adaptive-energy fallback, speech gating |
| [server/stt.py](server/stt.py) | Deepgram Nova-2 stream, faster-whisper fallback |
| [server/llm.py](server/llm.py) | Gemini 2.0 Flash / Groq streaming + tool loop |
| [server/tts.py](server/tts.py) | edge-tts, streaming MP3→PCM, sentence chunking |
| [server/tools.py](server/tools.py) | The three tools + JSON schemas |
| [server/session.py](server/session.py) | TTL session memory, sanitizing, call records |
| [web/](web/) | Next.js 15 UI: meters, transcript, tool log, latency |

## Run it

Backend (Python 3.11+):

```bash
cd server && pip install -r requirements.txt && python main.py
```

Frontend:

```bash
cd web && npm install && npm run dev
```

Open http://localhost:3000 and hit **Start call**. Grant the mic. You can also
type a turn and hit **Send** — it goes through the identical server pipeline,
which is the quickest way to test tool calling without talking.

Denying the mic does not end the call: the agent still speaks and typed turns
still work, which is how you test on a box with no microphone. **End call**
waits for the call record, so it takes a few seconds — that is an LLM round
trip writing the summary.

### Keys (all optional)

Copy `.env.example` to `.env` in the repo root:

| Key | Without it |
|---|---|
| `GROQ_API_KEY` | falls back to Gemini, then to the offline script |
| `GEMINI_API_KEY` | — |
| `DEEPGRAM_API_KEY` | falls back to local faster-whisper, then to no STT at all |

Groq is tried first, not Gemini. Measured on the free tiers: Groq returns its
first token in ~600 ms, Gemini in 3–5 s and it 503s under load, and the whole
turn has 1200 ms. `LLM_PROVIDER=gemini` flips the order back.

TTS needs no key, ever. With no STT the call still connects and typed turns
still work — the `ready` message reports which STT and LLM actually got picked.

`faster-whisper` and `numpy` are only needed for the local-STT path; nothing
else imports them.

### Speakers, not headphones

On speakers the agent hears itself and interrupts itself in a loop. Two
defences: the browser's own AEC is on, and while the agent's audio is still
playing the server ducks the VAD — louder speech, sustained longer, before it
counts as a barge-in — and stops feeding the microphone to the transcriber so
the agent can never answer its own voice. A caller who genuinely talks over it
still cuts in.

The room decides the numbers, so they are env knobs: `ECHO_THRESHOLD` (0.5
disables the duck, which is what you want on a headset), `ECHO_START_MS`,
`ECHO_TAIL_MS`.

## Wire protocol

`ws://localhost:8000/ws?session_id=<optional>`

Client sends raw PCM16 16 kHz mono frames (512 samples), plus
`{"type":"text","text":…}` for a typed turn and `{"type":"end"}` to hang up.

Server sends PCM16 16 kHz mono frames back, plus JSON:

| type | meaning |
|---|---|
| `ready` | session id, which STT and LLM got picked |
| `partial` / `final` | live and settled caller transcript |
| `assistant` | a sentence the agent is about to speak |
| `tool` | name, args and result of a tool call |
| `interrupt` | you barged in; drop any audio still queued |
| `metric` | `first_audio_ms` for the turn |
| `summary` | the full call record, sent at hangup |

## Tools

- `check_lead_qualification(email, company_size)` — deterministic scoring
  (business domain +30, headcount up to +50) into hot / warm / cold.
- `book_calendar_slot(email, datetime_iso)` — 30-minute slots, weekdays
  09:00–17:00 UTC, rejects the past and double-books.
- `lookup_kb(query)` — keyword search over [server/data/kb.json](server/data/kb.json).
  Returns "not found" rather than letting the model improvise.

Every call writes `server/data/calls/<session_id>.json`: transcript, tool
calls, lead facts, and an LLM-generated summary.

## Latency

Budget is 1200 ms end to end. The design choices that buy it:

- The reply is split into sentences and each one is sent to TTS as soon as it
  is complete, so audio starts before the model has finished thinking.
- MP3 from edge-tts is decoded incrementally, never buffered whole.
- The browser plays with 60 ms of jitter slack, not a full-clip wait.

Measured on this machine. Orchestration only, with providers mocked:

| stage | measured |
|---|---|
| edge-tts first audio, cold process | ~1350 ms |
| edge-tts first audio, warm | 360–550 ms |
| full turn, typed input → first PCM byte | 407–516 ms |

The cold number is DNS + TLS to Microsoft. `tts.prewarm()` runs at startup so a
caller never pays it.

Opening the socket to the first byte of the greeting: **330–450 ms**. The
greeting is spoken before the Deepgram handshake, not after it — that
handshake is most of a second and the caller has nothing to say yet.

On a spoken turn the transcriber's own endpoint decision ends the turn
immediately; `ENDPOINT_GRACE` is only spent when nothing endpointed for us.

Against the real providers (Groq + edge-tts, typed input → first PCM byte):

| turn | measured |
|---|---|
| plain reply, no tool | 1000–1220 ms |
| reply behind a tool call | 1400–1810 ms |

A tool turn is two LLM round trips, so it does not fit the budget and will not
without a faster model. A spoken turn also adds ~300 ms of Deepgram
endpointing on top of both numbers. Roughly 600 ms of every turn is Groq's
first token and 400 ms is edge-tts; both are the free tier's floor, so the next
real gain has to be bought, not coded.

The UI shows measured `first_audio_ms` per turn (silence detected → first byte
of audio out). `test_typed_turn_latency_under_budget` asserts the
orchestration overhead alone stays under budget with providers mocked.

## Tests

```bash
cd server && pytest
```

Run the one network test with `RUN_LIVE=1 pytest -k live` — it hits the real
edge-tts and asserts the decoded PCM is real audio, not silence.

Covers the tools, prompt sanitizing, session TTL and call records, VAD start
and end detection on synthetic audio, the echo guard opening and closing,
sentence chunking, and three websocket tests that drive a simulated audio
turn, a barge-in, and the latency benchmark against mocked providers.

## Known corners

- Session memory is a process-local dict. Swap `SessionStore` for Redis before
  running more than one worker.
- Leads, bookings and the KB are JSON files behind a process lock.
- KB search is term overlap, not embeddings.
- Silero VAD needs torch (~200 MB). Skip it and the adaptive energy gate takes
  over — fine for a headset, worse in a noisy room.
- Model names on the free tiers rot. `gemini-2.0-flash` and
  `llama-3.3-70b-versatile` were both retired during this build; the current
  defaults are `gemini-3.1-flash-lite` and `openai/gpt-oss-120b`. A 404 from a
  provider means the name moved again, not that the code broke.
- A provider that fails mid-turn costs the caller that turn ("Sorry, I lost
  that for a second"); there is no runtime failover between Groq and Gemini,
  only the choice made at startup.
- On Windows the `&` in this folder's name breaks `npx`; use
  `node ./node_modules/next/dist/bin/next dev` or rename the folder.
