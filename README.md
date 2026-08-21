# Voice AI Support & Qualification Agent

[![ci](https://github.com/natarovvv/voice-qualification-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/natarovvv/voice-qualification-agent/actions/workflows/ci.yml)

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
| [server/llm.py](server/llm.py) | Groq / Gemini streaming + tool loop, runtime failover |
| [server/tts.py](server/tts.py) | Deepgram Aura, edge-tts fallback, sentence chunking |
| [server/tools.py](server/tools.py) | The three tools + JSON schemas |
| [server/session.py](server/session.py) | TTL session memory, sanitizing, sealed call records |
| [server/metrics.py](server/metrics.py) | Prometheus counters and the latency histogram |
| [server/storage.py](server/storage.py) | Leads and bookings, sealed either way: JSON files or Postgres |
| [web/](web/) | Next.js 16 UI: meters, transcript, tool log, latency |
| [web/lib/voice.ts](web/lib/voice.ts) | Mic capture, jitter buffer, barge-in, session resume |

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

If the connection drops instead of ending — wifi blinks, a tunnel dies — the
page redials on its own and the agent picks the same conversation back up
rather than greeting you from scratch. Reload the tab and **Start call** does
the same thing by hand, because the session id is in `sessionStorage` either
way. That window is `RESUME_GRACE`, 60 seconds by default.

### Keys (all optional)

Copy `.env.example` to `.env` in the repo root:

| Key | Without it |
|---|---|
| `GROQ_API_KEY` | falls back to Gemini, then to the offline script |
| both LLM keys | with only one, a provider outage is a failed turn rather than a slower one |
| `GEMINI_API_KEY` | — |
| `DEEPGRAM_API_KEY` | STT falls back to local faster-whisper, then to none; the voice falls back to edge-tts |
| `CALL_ENCRYPTION_KEY` | records, leads and bookings are stored in the clear, as before — files and Postgres both |
| `REDIS_URL` | sessions live in a process-local dict instead of a shared store |
| `DATABASE_URL` | leads and bookings stay JSON files, which one process may write |

Groq is tried first, not Gemini. Measured on the free tiers: Groq returns its
first token in ~600 ms, Gemini in 3–5 s and it 503s under load, and the whole
turn has 1200 ms. `LLM_PROVIDER=gemini` flips the order back.

#### Failover

With both keys set the two cover each other during the call, not just at
startup. A turn whose provider throws is retried on the other one, and the
caller hears an answer instead of "sorry, I lost that for a second".

**A turn is never redone.** The reply is streamed, and a sentence is in the
caller's ear the moment it is complete — so once this turn has spoken a word or
run a tool, it belongs to the provider that started it. Handing it over would
repeat the sentence, or book the slot twice. Failing over is only safe before
either has happened, which is where providers fail anyway: a 429, a 503, or a
socket that never answers.

A provider that fails sits out `LLM_COOLDOWN` (default 30 s). Without that,
every turn of an outage pays the dead provider's timeout before falling
through, which is slower than having no failover at all. When the cooldown
expires the preferred provider gets the next turn back; if both are benched,
both get tried anyway, because a long shot beats a certain failure.

`LLM_STREAM_TIMEOUT` (default 8 s) is the read deadline while a reply is
streaming, separate from the 30 s the end-of-call summary keeps. The caller is
waiting on the first one, so it is what decides how quickly a hang becomes a
failover; Gemini's 3–5 s to first token is why it is not tighter.

Failovers are visible in `/metrics` as `voice_llm_errors_total{provider}` and
`voice_llm_failovers_total{to}`. A chain that is quietly limping shows up
there before it shows up in a call.

One Deepgram key covers both directions: Nova-2 transcribes and Aura speaks.
Without it the voice falls back to edge-tts, which is fine on a laptop and not
fine in front of a caller — see Security. With no STT the call still connects
and typed turns still work; the `ready` message reports which STT, TTS and LLM
actually got picked.

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

`{"type":"end"}` is the difference between hanging up and being cut off. A
socket that closes without it leaves the call resumable for `RESUME_GRACE`
seconds (default 60): reconnect with that `session_id` and the conversation
continues, history and all, greeted as a call that got cut off rather than a
new one — and the browser reconnects on its own, so in practice nobody has to.
If nobody comes back, the record is written when the grace expires.

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
- The Aura socket is opened once per call, not once per sentence. Connecting
  costs ~800 ms and is most of the latency of a one-shot request; paying it at
  call setup is what keeps a sentence at 235–484 ms.
- Aura returns linear16 at 16 kHz, so on that path there is no MP3 decode at
  all. edge-tts still decodes incrementally, never buffered whole.
- The browser plays with 60 ms of jitter slack, not a full-clip wait.

Measured on this machine:

| stage | deepgram-aura | edge-tts |
|---|---|---|
| greeting, first audio | 1172 ms | 437 ms |
| sentence mid-call, first audio | 235–484 ms | 360–550 ms |
| full turn, typed input → first PCM byte (mocked providers) | 407–516 ms | 407–516 ms |

Aura costs about 700 ms on the greeting and pays it back on every sentence
after. The greeting is the one sentence that genuinely waits for the socket,
because nothing can be said before the voice is connected. `tts.prewarm()`
warms DNS and TLS at startup so the per-call connect is the warm one.

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
of audio out), and the server keeps the same numbers in a histogram — see
below. `test_typed_turn_latency_under_budget` asserts the orchestration
overhead alone stays under budget with providers mocked.

## Observability

`GET /metrics` returns Prometheus text: how busy the box is, how many turns it
answered, how many of those failed, which tools were called, how often one LLM
provider had to cover for the other, and how long the caller waited to hear
something.

```bash
curl -H "Authorization: Bearer $AUTH_TOKEN" localhost:8000/metrics
```

It is behind the same token as `/data` whenever `AUTH_TOKEN` is set. Turn
counts and latency are exactly the operational detail `/health` withholds. With
no token the server binds loopback by default and shouts at startup if it does
not, so that case is already answered rather than answered twice.

`voice_first_audio_seconds` is a histogram whose boundaries bracket the 1200 ms
budget, so the share of turns inside it reads off a single bucket with no
scraper at all:

```bash
curl -s -H "Authorization: Bearer $AUTH_TOKEN" localhost:8000/metrics \
  | awk '/le="1.2"/ {ok=$2} /_seconds_count/ {print $2 ? ok/$2 : "no turns yet"}'
```

A histogram rather than a running p95, because percentiles do not average: an
in-process p95 stops meaning anything the moment there is a second worker,
while bucket counts add up across workers and the scraper takes the quantile
from the sum. `histogram_quantile(0.95, rate(voice_first_audio_seconds_bucket[5m]))`
is the p95 once something is scraping.

Only spoken turns are timed. A typed turn has no moment of silence to measure
from, so it is counted in `voice_turns_total` and left out of the histogram
rather than logged as a suspiciously fast one.

## Tests

```bash
cd server && pytest
cd web && npm test
```

Every push and pull request runs them on GitHub Actions
([.github/workflows/ci.yml](.github/workflows/ci.yml)), along with `next build`
for the web app. CI installs `requirements.lock` rather than
`requirements.txt`: the lock is the core path, and the extras would pull torch
and CUDA wheels for two fallbacks the tests stub out anyway. No keys are
configured there and none are needed — the live tests skip themselves and
`pgserver` brings its own PostgreSQL. `cryptography` is in the lock so the
sealed path is covered there too, not just where a key happens to be set.

Run the one network test with `RUN_LIVE=1 pytest -k live` — it hits the real
edge-tts and asserts the decoded PCM is real audio, not silence.

Covers the tools, prompt sanitizing, session TTL and call records, VAD start
and end detection on synthetic audio, the echo guard opening and closing,
sentence chunking, and websocket tests that drive a simulated audio turn, a
barge-in, the latency benchmark against mocked providers, and the access
checks: a foreign origin, a bad token, a rate-limited typed turn, and session
ids that would escape the calls directory. The Redis store is covered against
`fakeredis` - real command semantics, real constructor, but nothing here has
been run against a live `redis-server`. Resume is covered end to end: a socket
that dies writes no record, a reconnect gets the same session and its first
half back, an abandoned call is written out when the grace expires, and a
deliberate hangup is not resumable. The hold that makes that safe across
workers is seven more mutations, all caught: a timer that never asks whether it
still has the call, one that settles for any hold rather than its own, a
reconnect that leaves the hold where it is, a drop that arms a timer without
one, a hold that never reaches Redis, one that stays takeable after it is
taken, and `GETDEL` split back into `GET` then `DEL`. The metrics endpoint is covered by a real
call: a spoken turn moves the turn counter, the tool counter and the histogram
together, and the number the histogram holds is the same one the caller's
browser was sent. A turn that blows up is counted both as a turn and as a
failure, and a call turned away for capacity is counted as rejected and *not*
as served, so a busy box cannot flatter its own latency by dropping traffic.

The browser half is 33 Vitest tests over jsdom. They cover what the client
tells the browser to do rather than what it renders: chunks of the agent's
voice scheduled back to back behind one 60 ms jitter buffer and never into the
past, PCM16 decoded without clipping the negative rail, a barge-in dropping
the queue **and** rewinding the clock so the next sentence is not stuck behind
audio nobody will hear, the mic frames stopping at a closed socket, a denied
microphone still leaving a usable call, and the session id being kept across a
drop and forgotten on a real hangup. The capture worklet is tested through a
`postMessage` that genuinely detaches the transferred buffer, because that is
what turns a missing `.slice()` into silence. Redialling is twelve more
mutations on top, all caught: a drop that just ends the call, a hangup that
gets redialled anyway, 1008 retried like any other close, a redial asking for
the id Start asked for instead of the one the server minted, a second meter
loop left running, a backoff that never backs off and one that never stops
doubling — the ceiling is what keeps a late recovery inside the window — and
the page offering Start while the client is already redialling. Every one of
these was checked by breaking the code it covers: all thirteen of the original
mutations fail the suite too. The
metrics went through the same treatment - eleven mutations, all caught,
including the one that matters most quietly: `bisect_right` for `bisect_left`,
which moves a turn landing exactly on the 1200 ms budget from inside it to
outside. Failover is eleven tests and ten more mutations, all caught — that a
provider which already spoke keeps its turn, that one which already ran a tool
keeps it too, that a benched provider sits out the next turn and comes back
when the cooldown is up, and that a caller barging in is not mistaken for a
provider failing. Encryption is ten tests and nine more mutations: that the
transcript and the email are not in the file, that a tampered record is refused
rather than quietly returned, that a rotated key opens yesterday and seals
today, and that erasure reaches inside a sealed record — and counts the ones it
cannot open instead of passing over them. Sealing leads and bookings adds six
more: that a lead is not in the file, that a plain rows file re-seals itself on
the next write without losing the rows already in it, that the booking overlap
still holds through the seal, and — the one that would hurt — that a rows file
under a key we do not have is refused rather than read as empty and overwritten.

Postgres is covered against a real
PostgreSQL booted from the `pgserver` wheel, including six independent stores
racing for one slot; with the constraint dropped that test sells it six times.
Sealing it adds six tests and nine more mutations, all caught: that neither
the address nor its domain appears in any column of any row — `to_jsonb(t)`
takes the whole row, so a column added later is searched without anyone
remembering to — that the sealed `contact` still gives the address back, that
erasure finds a caller it cannot read and finds one written under a retired
key, that a row written before the key existed is still erasable, and that the
cap counts rows it cannot read while the overlap constraint, which never saw
an address in the first place, still holds. The last one builds the old table
by hand — `domain NOT NULL`, no `contact` — and checks that a sealed insert
lands in it and that the plaintext row already there is still erasable.

## Security

The defaults assume a laptop: the server binds `127.0.0.1` and auto-reload is
off unless `DEV=1`. Before it listens anywhere else:

- **Set `AUTH_TOKEN`** and the matching `NEXT_PUBLIC_WS_TOKEN`, and set
  `ALLOWED_ORIGINS` to your own origin. A websocket is exempt from the
  same-origin policy, so without these any page on the internet can open a
  call and spend your Deepgram and LLM budget. The browser token is a gate
  against other people's pages and against scanners, not against a determined
  caller — that needs a login this PRD does not have.
- **Terminate TLS and use `wss://`.** Over `ws://` the microphone audio, the
  transcript and the caller's email travel in clear text.
- Session ids are minted by the server (`secrets.token_urlsafe`) and never
  taken from the query string. A caller-chosen id is one it can guess, and
  guessing one lends it someone else's transcript and lead data.
- Limits per socket: audio at 4x realtime, 10 typed turns per 10 s, and
  `MAX_CALLS` sockets for the whole process.

Data and third parties, which are decisions rather than settings:

- **Encryption at rest:** set `CALL_ENCRYPTION_KEY` and every file this
  server writes under `server/data/` is sealed with Fernet — call records
  (transcript, email, summary) and, on the file backend, `leads.json` and
  `bookings.json`. Generate a key with:

  ```bash
  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
  ```

  The setting takes a comma-separated list so a key can be rotated without a
  migration: the **first** key seals, any of them opens. Files written before
  the key existed are still plain JSON and still read, so turning it on is not
  a migration either — both kinds share the directory. Leads and bookings go
  further and re-seal themselves under the current key the next time a row is
  added, because unlike a call record they are rewritten; a call record is
  written once at the end of the call and never touched again.

  The session id stays legible, because it is the filename already and
  retention has to find a file without opening it. It is a random token, not
  anything about the caller. Nothing is kept legible in the rows files: they
  are found by their own name.

  A rows file that cannot be opened — sealed under a key this process does not
  have — raises rather than reading back as "no rows", which would let the very
  next lead overwrite everything in it. The agent reports a tool failure it can
  talk about, and the file is left exactly as it was.

  Setting the key with `cryptography` missing, or setting a key that is not a
  key, stops the process from starting. Writing plaintext you believe is
  encrypted is worse than not starting.

  **Postgres is sealed by the same key.** The address is no longer a column:
  the row carries an HMAC of it under those same keys, and the address itself
  sits sealed beside it in `contact`. Equality is all the lookups ever needed —
  the booking cap and erasure are both `WHERE email = ANY(...)` — so the index
  still works, and rotation works the way sealing does, because a write goes
  under the first key while a lookup goes out under all of them. The `domain`
  column is gone: it was `email.split("@")[1]`, which made it a second copy of
  the same personal datum sitting in the clear.

  Booking times stay legible, because `EXCLUDE USING gist` is the entire reason
  that table is not a file and a range index cannot read ciphertext. An empty
  slot is not personal data; who is in it is, and that part is sealed.

  **What this does not cover: rows already in Postgres.** A file re-seals
  itself because it is rewritten whole; a row is not, so anything written
  before the key was set stays in the clear until something rewrites it.
  Erasure still reaches those rows — the plain address goes into the lookup
  beside the hashes — but re-sealing them is an `UPDATE` this repo does not
  ship. Turning the key on is still not a migration; it is just not a
  retrofit either.

  `CALL_RETENTION_DAYS` (default 30) deletes records at startup whether sealed
  or not, and the erasure endpoint below handles a named caller.
- Audio goes to Deepgram, text goes to Groq or Gemini. With `DEEPGRAM_API_KEY`
  set, the voice is Deepgram Aura — the same vendor and contract as the
  transcriber, so one DPA covers both directions. Without the key it falls back
  to `edge-tts`, an undocumented consumer Microsoft endpoint with no commercial
  agreement behind it; the server logs a warning when it does. Keep that path
  for development only, and get DPAs with Deepgram and your LLM provider before
  a live caller's voice reaches either.
- **Metrics:** `GET /metrics` is behind `Authorization: Bearer $AUTH_TOKEN`
  when one is set. Call volume and latency are not caller data, but they are a
  free read on how much traffic you carry and when you are struggling.
- **Erasure requests:** `DELETE /data?email=...` with `Authorization: Bearer
  $AUTH_TOKEN` deletes that caller's leads, bookings and call records and
  reports what it removed. It is refused outright unless `AUTH_TOKEN` is set,
  and the agent cannot reach it — an erase tool the model could call would let
  a caller delete someone else's records by naming their address. A call where
  the caller never gave an email has no key to match on; the retention window
  is what clears those. A record it cannot open — sealed with a key this
  deployment does not have — is counted in `removed["unreadable"]` rather than
  passed over quietly: a record nobody can read is a record nobody can prove
  was erased, and the operator answering the request needs to see the count is
  short.

## Known corners

- Session memory is a process-local dict unless `REDIS_URL` is set, in which
  case sessions are shared and expire on Redis's own TTL. A call still lives on
  one worker for its whole life; what is shared is where it can be picked back
  up. When a socket dies, the worker puts a *hold* on the call — a key with the
  worker's own token — and arms a local timer to write the record when the
  grace expires. Whoever the reconnect lands on takes that hold, atomically,
  and the timer that finds the hold gone writes nothing: its copy of the call
  stopped growing when the socket died, and the caller is elsewhere adding to a
  newer one. Sessions the worker is serving stay in a local dict too, holds
  included, so a Redis outage degrades to the single-process behaviour instead
  of dropping the caller's history mid-sentence.
- A dropped call redials itself. The browser keeps the session id in
  `sessionStorage` and hands it back, and the client reopens the socket on its
  own — half a second later, then one, then two, capped at five, for as long as
  the server still holds the call. Only the socket is rebuilt: the microphone,
  the AudioContext and the worklet stay up, so nobody is asked for permission
  twice. It gives up after `RESUME_GRACE`, because past that the record is
  written and there is nothing left to rejoin. Two things stop it early: the
  caller pressing End, and close code 1008 — a bad token, or an audio rate
  limit, neither of which a reconnect improves.
- Leads and bookings live in Postgres when `DATABASE_URL` is set and in JSON
  files when it is not. The files are written temp-file-then-rename so a crash
  cannot truncate one, but their `threading.Lock` is process-local and does
  nothing across processes: on files, **two workers will sell the same slot
  twice.** On Postgres they cannot, because the overlap is an `EXCLUDE USING
  gist` constraint rather than an application check — the database refuses the
  second booking whoever asks. With `DATABASE_URL` and `REDIS_URL` both set,
  more than one worker is safe.
- Metrics are per-process and start at zero on restart, which is the normal
  shape for a scrape endpoint — the scraper keeps the history. It does mean a
  single `curl` behind a load balancer reads one worker's tally, not the fleet's.
  Counters and histogram buckets both sum across workers, so a scraper hitting
  every replica gets the real number; a human with `curl` does not.
- The LLM cooldown is one failure wide: a single 503 benches a provider for
  30 s rather than counting failures first. That is deliberate on a free tier
  where a 503 usually means the next request fails too, but it does mean one
  unlucky request moves the call to the slower model for half a minute. Count
  failures before benching if you move to a paid tier where a blip is a blip.
- The session copy in Redis is not sealed. It holds the live transcript for
  `SESSION_TTL` (30 minutes) so a dropped call can resume, and Redis is
  normally memory-only; the 30-day copy on disk is the one worth a key. Turn
  off Redis persistence, or encrypt its volume, if that is not good enough.
- The knowledge base stays a file either way. It is content that ships with the
  repo, not caller data.
- The per-email booking cap is checked inside the transaction but under read
  committed, so two simultaneous bookings can land at cap + 1. It is a spam
  guard rather than money; the overlap constraint is the invariant that
  actually holds.
- KB search is term overlap, not embeddings.
- edge-tts remains the keyless fallback and is development-only; see Security.
- Silero VAD needs torch (~200 MB). Skip it and the adaptive energy gate takes
  over — fine for a headset, worse in a noisy room.
- Model names on the free tiers rot. `gemini-2.0-flash` and
  `llama-3.3-70b-versatile` were both retired during this build; the current
  defaults are `gemini-3.1-flash-lite` and `openai/gpt-oss-120b`. A 404 from a
  provider means the name moved again, not that the code broke.
- On Windows the `&` in this folder's name breaks `npx`; use
  `node ./node_modules/next/dist/bin/next dev` or rename the folder.
