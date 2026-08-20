"""FastAPI websocket server: the voice loop.

One websocket per call. Client sends PCM16 16 kHz mono frames, server sends
PCM16 16 kHz mono back plus JSON control messages.

    client -> server   binary PCM | {"type":"text","text":...} | {"type":"end"}
    server -> client   binary PCM | {"type": ready|partial|final|assistant
                                             |tool|interrupt|metric|summary|error}
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import llm as llm_mod
import tools
import tts
from config import (
    ALLOWED_ORIGINS,
    AUTH_TOKEN,
    BYTES_PER_SEC,
    ECHO_TAIL,
    HOST,
    MAX_CALLS,
    MAX_TEXT_TURNS,
    MAX_TURN_CHARS,
    PORT,
    RATE_LIMIT_FACTOR,
    SYSTEM_PROMPT,
    TEXT_WINDOW,
)
from session import STORE, purge_old_calls, sanitize
from stt import make_stt
from vad import SpeechGate, pcm_seconds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("voice")

GREETING = "Hi, you have reached support. I am Aria. What can I help you with today?"
ENDPOINT_GRACE = 0.35  # wait this long after silence for a straggling transcript


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient()
    app.state.llm = llm_mod.make_llm(app.state.http)
    app.state.calls = 0
    asyncio.create_task(tts.prewarm())  # pay the TLS handshake before a caller does
    log.info("llm provider: %s | sessions: %s", app.state.llm.name, STORE.name)
    log.info("retention: purged %s expired call records", purge_old_calls())
    if not AUTH_TOKEN and HOST not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "SERVING %s WITHOUT AUTH_TOKEN: anyone who can reach this port can "
            "spend your STT and LLM budget", HOST
        )
    yield
    await app.state.http.aclose()


app = FastAPI(title="Voice AI Support & Qualification Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
async def health() -> dict:
    # Which model you run and how busy you are is nobody's business but yours.
    return {"ok": True}


@app.delete("/data")
async def erase(email: str, request: Request) -> dict:
    """Erase everything held about one caller - the right-to-erasure handle.

    Operator-only, and refused outright unless AUTH_TOKEN is configured: an
    open delete endpoint is a worse problem than the one it solves. Erasure
    requests arrive by mail or through support, and a human runs this; the
    agent cannot reach it, so no amount of talking to it deletes anything.
    """
    if not AUTH_TOKEN:
        raise HTTPException(503, "set AUTH_TOKEN to enable erasure requests")
    supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not secrets.compare_digest(supplied, AUTH_TOKEN):
        raise HTTPException(401, "bad token")

    result = await asyncio.to_thread(tools.erase_caller, email)
    if not result["ok"]:
        raise HTTPException(400, result["error"])
    # The address is the thing being erased, so it does not go in the log.
    log.info("erasure request completed: %s", result["removed"])
    return result


def authorized(ws: WebSocket) -> bool:
    """Gate the websocket by origin and, if configured, a shared secret.

    A websocket is exempt from the same-origin policy: without this check any
    page on the internet can open a call on your bill. A non-browser client
    sends no Origin at all, so that case falls through to the token.
    """
    origin = ws.headers.get("origin")
    if origin is not None and "*" not in ALLOWED_ORIGINS and origin not in ALLOWED_ORIGINS:
        log.warning("rejected websocket from origin %s", origin)
        return False
    if AUTH_TOKEN and not secrets.compare_digest(ws.query_params.get("token", ""), AUTH_TOKEN):
        log.warning("rejected websocket with a bad token")
        return False
    return True


class RateLimiter:
    """Budget per rolling window, so one socket cannot flood the box."""

    def __init__(self, budget: int, window: float = 1.0) -> None:
        self.budget, self.window = budget, window
        self._start = time.monotonic()
        self._used = 0

    def allow(self, n: int = 1) -> bool:
        now = time.monotonic()
        if now - self._start >= self.window:
            self._start, self._used = now, 0
        self._used += n
        return self._used <= self.budget


class Call:
    def __init__(self, ws: WebSocket, session, llm) -> None:
        self.ws, self.session, self.llm = ws, session, llm
        self.stt = make_stt()
        self.tts = tts.make_tts()
        self.gate = SpeechGate()
        self.audio_limit = RateLimiter(BYTES_PER_SEC * RATE_LIMIT_FACTOR)
        # Typed turns skip the microphone and so skip the audio budget: without
        # their own limit one socket can drive the LLM in a tight loop.
        self.text_limit = RateLimiter(MAX_TEXT_TURNS, TEXT_WINDOW)
        self.speaking: asyncio.Task | None = None
        self.finisher: asyncio.Task | None = None
        self.pending: list[str] = []
        self.turn_start = 0.0
        self.play_until = 0.0  # monotonic time the room stops hearing our own voice
        self.closed = False

    # ---------- outbound ----------

    # A caller who hangs up mid-sentence leaves a send in flight, and starlette
    # reports that as WebSocketDisconnect, RuntimeError or anyio's
    # ClosedResourceError depending on where it was caught. Naming them is how
    # one of them gets missed, and a missed one kills the turn that still has
    # to write the call record. Any failed send means the same thing: gone.
    # CancelledError is a BaseException, so barge-in still passes through.

    async def send(self, **msg) -> None:
        if not self.closed:
            try:
                await self.ws.send_text(json.dumps(msg))
            except Exception:  # noqa: BLE001
                self.closed = True

    async def send_audio(self, pcm: bytes) -> None:
        if not self.closed:
            try:
                await self.ws.send_bytes(pcm)
                # the client is still playing this out after the last byte leaves
                self.play_until = max(self.play_until, time.monotonic()) + pcm_seconds(pcm)
            except Exception:  # noqa: BLE001
                self.closed = True

    # ---------- the reply half of a turn ----------

    def _system(self) -> str:
        facts = self.session.facts
        if not facts:
            return SYSTEM_PROMPT
        return f"{SYSTEM_PROMPT}\nAlready established this call: {json.dumps(facts)}"

    async def _on_tool(self, name: str, args: dict, result: dict) -> None:
        self.session.add_tool_call(name, args, result)
        await self.send(type="tool", name=name, args=args, result=result)

    async def say(self, text: str, first: bool = False) -> None:
        """Speak one sentence and report time-to-first-audio."""
        await self.send(type="assistant", text=text)
        async for pcm in self.tts.speak(text):
            if first and self.turn_start:
                await self.send(
                    type="metric",
                    first_audio_ms=round((time.monotonic() - self.turn_start) * 1000),
                )
                self.turn_start = 0.0  # a stale one would time the whole last turn
                first = False
            await self.send_audio(pcm)

    async def respond(self, user_text: str) -> None:
        self.session.add_turn("user", user_text)
        await self.send(type="final", speaker="caller", text=user_text)
        buffer, spoken, first = "", [], True
        try:
            async for delta in self.llm.stream(self._system(), self.session.history, self._on_tool):
                buffer += delta
                sentences, buffer = tts.split_sentences(buffer)
                for s in sentences:
                    await self.say(s, first=first)
                    spoken.append(s)
                    first = False
            tail, _ = tts.split_sentences(buffer, flush=True)
            for s in tail:
                await self.say(s, first=first)
                spoken.append(s)
                first = False
        except asyncio.CancelledError:
            log.info("turn interrupted by caller")
            raise
        except Exception:  # noqa: BLE001 - one bad turn must not drop the call
            # The provider's exception text carries its URL, model and response
            # body. That belongs in the log, not on the caller's socket.
            log.exception("turn failed")
            await self.send(type="error", message="the assistant hit an error")
            await self.say("Sorry, I lost that for a second. Could you say it again?")
        finally:
            if spoken:
                self.session.add_turn("assistant", " ".join(spoken))
            # Persist the finished turn. A turn the caller talked over is
            # cancelled here and skips this; the next one writes it anyway.
            await STORE.put(self.session)

    # ---------- turn taking ----------

    async def interrupt(self) -> None:
        if self.speaking and not self.speaking.done():
            self.speaking.cancel()
            self.play_until = 0.0  # the client drops its queue on this message
            await self.send(type="interrupt")  # the caller stops hearing us here
            # ...and only then wait for the turn to unwind and the voice to
            # drop its in-flight audio. Both touch state the next turn needs,
            # and neither is something the caller is waiting on.
            # However that turn ended is its own business - respond() already
            # logged it. Letting it out here would take the call record with it.
            with suppress(asyncio.CancelledError, Exception):
                await self.speaking
            await self.tts.reset()

    async def _finish_turn(self, grace: float) -> None:
        """Fire after the grace period; whatever transcript arrived is the turn."""
        await asyncio.sleep(grace)
        text = sanitize(" ".join(self.pending).strip())
        self.pending.clear()
        if text:
            self.speaking = asyncio.create_task(self.respond(text))

    def schedule_turn(self, grace: float = ENDPOINT_GRACE) -> None:
        if self.finisher and not self.finisher.done():
            self.finisher.cancel()
        self.finisher = asyncio.create_task(self._finish_turn(grace))

    async def on_audio(self, pcm: bytes) -> None:
        if not self.audio_limit.allow(len(pcm)):
            await self.send(type="error", message="audio rate limit exceeded")
            raise WebSocketDisconnect(code=1008)
        self.gate.echo = time.monotonic() < self.play_until + ECHO_TAIL
        for event in self.gate.update(pcm):
            if event == "start":
                if self.finisher and not self.finisher.done():
                    self.finisher.cancel()
                await self.interrupt()
            else:  # "end"
                # whichever endpoints first starts the clock - Deepgram's own
                # decision often beats our VAD, and it must not go unmeasured
                self.turn_start = self.turn_start or time.monotonic()
                await self.stt.endpoint()
                self.schedule_turn()
        if self.gate.echo and not self.gate.active:
            # ponytail: drops the first ~ECHO_START_MS of a real barge-in too.
            # Cheaper than a hold buffer, and the caller keeps talking anyway.
            await self.stt.keepalive()
            return  # our own voice: transcribing it would make the agent answer itself
        await self.stt.send(pcm)

    async def stt_loop(self) -> None:
        while True:
            event = await self.stt.events.get()
            if event["type"] == "partial":
                await self.send(type="partial", text=event["text"])
            else:
                self.pending.append(event["text"])
                if event.get("ended"):
                    self.turn_start = self.turn_start or time.monotonic()
                    # the transcriber endpointed the utterance itself; waiting
                    # out our own grace on top of that is pure added latency
                    self.schedule_turn(grace=0)
                elif not self.gate.active:  # silence already detected: go now
                    self.schedule_turn()

    async def hangup(self):
        await self.interrupt()
        await self.stt.close()
        await self.tts.close()
        if not self.session.transcript:
            return
        summary = await llm_mod.summarize(self.llm, self.session.transcript, self.session.facts)
        record = self.session.save(summary)
        await STORE.drop(self.session.id)  # the record is on disk now
        log.info("call %s saved: %s turns", self.session.id, len(record["transcript"]))
        return record


@app.websocket("/ws")
async def voice_ws(ws: WebSocket) -> None:
    if not authorized(ws):
        await ws.close(code=1008)
        return
    if app.state.calls >= MAX_CALLS:
        log.warning("refused a call: %s already live", app.state.calls)
        await ws.close(code=1013)  # try again later
        return
    app.state.calls += 1
    await ws.accept()
    session = await STORE.get(ws.query_params.get("session_id"))
    call = Call(ws, session, app.state.llm)
    await call.send(
        type="ready",
        session_id=session.id,
        stt=call.stt.name,
        tts=call.tts.name,
        llm=call.llm.name,
        sample_rate=16000,
    )
    reader = asyncio.create_task(call.stt_loop())
    # The voice has to be connected before anything can be said, so this
    # handshake is the one the greeting genuinely waits on.
    try:
        await call.tts.start()
    except Exception:  # noqa: BLE001 - a mute agent still beats a refused call
        log.exception("tts failed to start; the call runs without audio")
    # Greet before connecting the transcriber, not after: that handshake is
    # most of a second and the caller cannot say anything worth hearing until
    # the greeting is out. Audio arriving early is dropped by an unstarted
    # stream, which costs nothing.
    call.speaking = asyncio.create_task(call.say(GREETING))
    call.session.add_turn("assistant", GREETING)
    try:
        await call.stt.start()
    except Exception:  # noqa: BLE001 - a dead transcriber must not refuse the call
        log.exception("stt failed to start; typed turns still work")

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                await call.on_audio(data)
            elif (text := msg.get("text")) is not None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue  # a malformed frame is not a reason to drop the call
                kind = payload.get("type")
                if kind == "text":  # typed turn: same pipeline, no microphone
                    if not call.text_limit.allow():
                        await call.send(type="error", message="too many turns; slow down")
                        break
                    await call.interrupt()
                    text = str(payload.get("text") or "")[:MAX_TURN_CHARS]
                    call.speaking = asyncio.create_task(call.respond(text))
                elif kind == "end":
                    break
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.exception("websocket failed")
    finally:
        app.state.calls -= 1
        reader.cancel()
        record = await call.hangup()
        if record:
            await call.send(type="summary", record=record)
        try:
            await ws.close()
        except RuntimeError:
            pass


if __name__ == "__main__":
    import uvicorn

    # Loopback and no auto-reload by default: reload watches the tree and forks
    # a child, which is a development convenience, not a thing to expose.
    uvicorn.run("main:app", host=HOST, port=PORT, reload=bool(os.getenv("DEV")))
