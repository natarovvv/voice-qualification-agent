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
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import llm as llm_mod
import tts
from config import BYTES_PER_SEC, RATE_LIMIT_FACTOR, SYSTEM_PROMPT
from session import STORE, sanitize
from stt import make_stt
from vad import SpeechGate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("voice")

GREETING = "Hi, you have reached support. I am Aria. What can I help you with today?"
ENDPOINT_GRACE = 0.35  # wait this long after silence for a straggling transcript


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient()
    app.state.llm = llm_mod.make_llm(app.state.http)
    asyncio.create_task(tts.prewarm())  # pay the TLS handshake before a caller does
    log.info("llm provider: %s", app.state.llm.name)
    yield
    await app.state.http.aclose()


app = FastAPI(title="Voice AI Support & Qualification Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "llm": app.state.llm.name, "sessions": len(STORE)}


class RateLimiter:
    """Cap inbound audio at N x realtime so one socket cannot flood the box."""

    def __init__(self, factor: int = RATE_LIMIT_FACTOR) -> None:
        self.budget = BYTES_PER_SEC * factor
        self._window = time.monotonic()
        self._bytes = 0

    def allow(self, n: int) -> bool:
        now = time.monotonic()
        if now - self._window >= 1.0:
            self._window, self._bytes = now, 0
        self._bytes += n
        return self._bytes <= self.budget


class Call:
    def __init__(self, ws: WebSocket, session, llm) -> None:
        self.ws, self.session, self.llm = ws, session, llm
        self.stt = make_stt()
        self.gate = SpeechGate()
        self.limiter = RateLimiter()
        self.speaking: asyncio.Task | None = None
        self.finisher: asyncio.Task | None = None
        self.pending: list[str] = []
        self.turn_start = 0.0
        self.closed = False

    # ---------- outbound ----------

    async def send(self, **msg) -> None:
        if not self.closed:
            try:
                await self.ws.send_text(json.dumps(msg))
            except (WebSocketDisconnect, RuntimeError):
                self.closed = True

    async def send_audio(self, pcm: bytes) -> None:
        if not self.closed:
            try:
                await self.ws.send_bytes(pcm)
            except (WebSocketDisconnect, RuntimeError):
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
        async for pcm in tts.speak(text):
            if first and self.turn_start:
                await self.send(
                    type="metric",
                    first_audio_ms=round((time.monotonic() - self.turn_start) * 1000),
                )
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
        except Exception as exc:  # noqa: BLE001 - one bad turn must not drop the call
            log.exception("turn failed")
            await self.send(type="error", message=str(exc))
            await self.say("Sorry, I lost that for a second. Could you say it again?")
        finally:
            if spoken:
                self.session.add_turn("assistant", " ".join(spoken))

    # ---------- turn taking ----------

    async def interrupt(self) -> None:
        if self.speaking and not self.speaking.done():
            self.speaking.cancel()
            await self.send(type="interrupt")

    async def _finish_turn(self) -> None:
        """Fire after the grace period; whatever transcript arrived is the turn."""
        await asyncio.sleep(ENDPOINT_GRACE)
        text = sanitize(" ".join(self.pending).strip())
        self.pending.clear()
        if text:
            self.speaking = asyncio.create_task(self.respond(text))

    def schedule_turn(self) -> None:
        if self.finisher and not self.finisher.done():
            self.finisher.cancel()
        self.finisher = asyncio.create_task(self._finish_turn())

    async def on_audio(self, pcm: bytes) -> None:
        if not self.limiter.allow(len(pcm)):
            await self.send(type="error", message="audio rate limit exceeded")
            raise WebSocketDisconnect(code=1008)
        for event in self.gate.update(pcm):
            if event == "start":
                if self.finisher and not self.finisher.done():
                    self.finisher.cancel()
                await self.interrupt()
            else:  # "end"
                self.turn_start = time.monotonic()
                await self.stt.endpoint()
                self.schedule_turn()
        await self.stt.send(pcm)

    async def stt_loop(self) -> None:
        while True:
            event = await self.stt.events.get()
            if event["type"] == "partial":
                await self.send(type="partial", text=event["text"])
            else:
                self.pending.append(event["text"])
                if not self.gate.active:  # silence already detected: go now
                    self.schedule_turn()

    async def hangup(self):
        await self.interrupt()
        await self.stt.close()
        if not self.session.transcript:
            return
        summary = await llm_mod.summarize(self.llm, self.session.transcript, self.session.facts)
        record = self.session.save(summary)
        log.info("call %s saved: %s turns", self.session.id, len(record["transcript"]))
        return record


@app.websocket("/ws")
async def voice_ws(ws: WebSocket) -> None:
    await ws.accept()
    session = STORE.get(ws.query_params.get("session_id"))
    call = Call(ws, session, app.state.llm)
    await call.stt.start()
    await call.send(
        type="ready", session_id=session.id, stt=call.stt.name, llm=call.llm.name, sample_rate=16000
    )
    reader = asyncio.create_task(call.stt_loop())
    call.speaking = asyncio.create_task(call.say(GREETING))
    call.session.add_turn("assistant", GREETING)

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                await call.on_audio(data)
            elif (text := msg.get("text")) is not None:
                payload = json.loads(text)
                kind = payload.get("type")
                if kind == "text":  # typed turn: same pipeline, no microphone
                    await call.interrupt()
                    call.speaking = asyncio.create_task(call.respond(payload.get("text", "")))
                elif kind == "end":
                    break
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.exception("websocket failed")
    finally:
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

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
