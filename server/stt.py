"""Speech to text.

Deepgram Nova-2 over a raw websocket when a key is present (streaming,
interim results, its own endpointing); faster-whisper on the local machine
otherwise, transcribing one utterance at a time on the VAD endpoint.

Both put ``{"type": "partial"|"final", "text": ...}`` on ``events``.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import time

from config import DEEPGRAM_API_KEY, SAMPLE_RATE, WHISPER_MODEL

log = logging.getLogger(__name__)

DG_URL = (
    "wss://api.deepgram.com/v1/listen"
    f"?encoding=linear16&sample_rate={SAMPLE_RATE}&channels=1"
    "&model=nova-2&language=en-US&punctuate=true&smart_format=true"
    "&interim_results=true&endpointing=300&vad_events=true"
)
MAX_UTTERANCE_SEC = 30


class DeepgramSTT:
    name = "deepgram"
    KEEPALIVE_SEC = 5  # Deepgram closes the socket after ~10 s of nothing

    def __init__(self, api_key: str = DEEPGRAM_API_KEY) -> None:
        self.api_key = api_key
        self.events: asyncio.Queue = asyncio.Queue()
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._last_sent = 0.0

    async def start(self) -> None:
        import websockets

        headers = {"Authorization": f"Token {self.api_key}"}
        try:  # websockets >= 13 renamed the kwarg
            self._ws = await websockets.connect(DG_URL, additional_headers=headers)
        except TypeError:
            self._ws = await websockets.connect(DG_URL, extra_headers=headers)
        self._reader = asyncio.create_task(self._read())
        log.info("STT: deepgram nova-2 connected")

    async def _read(self) -> None:
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                if msg.get("type") != "Results":
                    continue
                alt = msg["channel"]["alternatives"][0]
                text = (alt.get("transcript") or "").strip()
                if not text:
                    continue
                # speech_final is Deepgram's own endpoint decision; is_final
                # only settles a segment and more may still be coming.
                ended = bool(msg.get("speech_final"))
                final = ended or bool(msg.get("is_final"))
                await self.events.put(
                    {"type": "final" if final else "partial", "text": text, "ended": ended}
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - connection drops end the stream
            log.warning("deepgram reader stopped: %s", exc)

    async def send(self, pcm: bytes) -> None:
        if self._ws:
            self._last_sent = time.monotonic()
            await self._ws.send(pcm)

    async def keepalive(self) -> None:
        """Hold the socket open while we are deliberately not sending audio -
        the echo guard withholds the microphone for as long as the agent talks,
        which is easily past Deepgram's idle timeout."""
        if self._ws and time.monotonic() - self._last_sent > self.KEEPALIVE_SEC:
            self._last_sent = time.monotonic()
            await self._ws.send(json.dumps({"type": "KeepAlive"}))

    async def endpoint(self) -> None:
        """Deepgram does its own endpointing."""

    async def close(self) -> None:
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:  # noqa: BLE001
                pass
            await self._ws.close()
        if self._reader:
            self._reader.cancel()


class WhisperSTT:
    """Local fallback. Buffers an utterance, transcribes it on the VAD endpoint."""

    name = "faster-whisper"
    _model = None  # class-level: loading it twice would double the RAM

    def __init__(self, model_size: str = WHISPER_MODEL) -> None:
        self.model_size = model_size
        self.events: asyncio.Queue = asyncio.Queue()
        self._buf = bytearray()
        self._busy = False

    async def start(self) -> None:
        if WhisperSTT._model is None:
            from faster_whisper import WhisperModel

            WhisperSTT._model = await asyncio.to_thread(
                WhisperModel, self.model_size, device="cpu", compute_type="int8"
            )
        log.info("STT: faster-whisper %s ready", self.model_size)

    async def send(self, pcm: bytes) -> None:
        self._buf.extend(pcm)
        limit = MAX_UTTERANCE_SEC * SAMPLE_RATE * 2
        if len(self._buf) > limit:
            del self._buf[: len(self._buf) - limit]

    async def keepalive(self) -> None:
        """Local model, nothing to keep alive."""

    async def endpoint(self) -> None:
        audio, self._buf = bytes(self._buf), bytearray()
        if self._busy or len(audio) < SAMPLE_RATE:  # under ~0.5 s is a cough
            return
        self._busy = True
        try:
            text = await asyncio.to_thread(self._transcribe, audio)
            if text:
                # it only runs on the VAD endpoint, so the utterance is over
                await self.events.put({"type": "final", "text": text, "ended": True})
        finally:
            self._busy = False

    def _transcribe(self, audio: bytes) -> str:
        import numpy as np

        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = WhisperSTT._model.transcribe(samples, language="en", beam_size=1, vad_filter=False)
        return " ".join(s.text.strip() for s in segments).strip()

    async def close(self) -> None:
        self._buf.clear()


class NullSTT:
    """No key and no local model: the call still connects, and typed turns
    still work. Better than refusing the websocket."""

    name = "none"

    def __init__(self) -> None:
        self.events: asyncio.Queue = asyncio.Queue()

    async def start(self):
        log.warning("STT: none available - set DEEPGRAM_API_KEY or pip install faster-whisper")

    async def send(self, pcm: bytes):
        pass

    async def keepalive(self):
        pass

    async def endpoint(self):
        pass

    async def close(self):
        pass


def make_stt():
    if DEEPGRAM_API_KEY:
        return DeepgramSTT()
    if importlib.util.find_spec("faster_whisper"):
        return WhisperSTT()
    return NullSTT()
