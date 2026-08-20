"""Text to speech -> PCM 16 kHz 16-bit mono.

Two providers, same shape as ``make_stt``: Deepgram Aura over a websocket when
a key is present, edge-tts otherwise.

Aura is the one to run in front of real callers. It is the same vendor and the
same key as the transcriber, so it is covered by a real contract; edge-tts is
an undocumented consumer Microsoft endpoint with nothing behind it, which is
fine for a laptop and not fine for someone's voice.

Aura also speaks linear16 at our own sample rate, so the whole MP3 decode
disappears on that path. The decoder below is edge-tts's alone.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import aclosing
from typing import AsyncIterator

from config import DEEPGRAM_API_KEY, DEEPGRAM_TTS_MODEL, SAMPLE_RATE, TTS_VOICE

log = logging.getLogger(__name__)

# Speak a sentence as soon as it is complete instead of waiting for the full
# reply - this is most of the difference between 400 ms and 2 s to first audio.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|[\n\r]+")
SOFT_LIMIT = 140  # flush a long clause even without punctuation


class Mp3ToPcm:
    """Streaming MP3 decoder. Feed bytes, get PCM16 mono at SAMPLE_RATE."""

    def __init__(self, rate: int = SAMPLE_RATE) -> None:
        import av

        self._ctx = av.CodecContext.create("mp3", "r")
        self._resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)

    @staticmethod
    def _pcm(frames) -> bytes:
        # plane 0 of a packed s16 mono frame is the samples; slicing to
        # samples*2 drops the codec's row padding. Verified byte-identical to
        # to_ndarray().tobytes(), and keeps numpy off the audio path.
        return b"".join(bytes(f.planes[0])[: f.samples * 2] for f in frames)

    def _resample(self, frames) -> bytes:
        return b"".join(self._pcm(self._resampler.resample(f)) for f in frames)

    def feed(self, chunk: bytes) -> bytes:
        out = bytearray()
        for packet in self._ctx.parse(chunk):
            out.extend(self._resample(self._ctx.decode(packet)))
        return bytes(out)

    def flush(self) -> bytes:
        """Drain the codec, then the resampler. The resampler's own tail is
        already at the output format - do not send it back through."""
        out = bytearray(self._resample(self._ctx.decode(None)))
        out.extend(self._pcm(self._resampler.resample(None)))
        return bytes(out)


def split_sentences(buffer: str, flush: bool = False) -> tuple[list[str], str]:
    """Pull complete sentences out of a growing buffer.

    Returns (sentences, leftover). With ``flush`` the leftover comes out too.
    """
    parts = _SENTENCE_END.split(buffer)
    leftover = parts.pop() if parts else ""
    out = [p.strip() for p in parts if p.strip()]
    if flush and leftover.strip():
        out.append(leftover.strip())
        leftover = ""
    elif len(leftover) > SOFT_LIMIT and " " in leftover:
        head, _, leftover = leftover.rpartition(" ")
        if head.strip():
            out.append(head.strip())
    return out, leftover


def _clean(text: str) -> str:
    """Strip anything a model writes that a voice should not read aloud."""
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


class DeepgramTTS:
    """Aura over one websocket held open for the whole call.

    Connecting costs ~800 ms, which is most of the latency of a one-shot
    request. Paying it once at call setup puts a sentence at ~250-450 ms to
    first audio; paying it per sentence would blow the turn budget on its own.
    """

    name = "deepgram-aura"
    DRAIN_TIMEOUT = 3.0  # a dead socket must not hang the call

    def __init__(self, api_key: str = DEEPGRAM_API_KEY, model: str = DEEPGRAM_TTS_MODEL) -> None:
        self.api_key, self.model = api_key, model
        self._ws = None

    @property
    def _url(self) -> str:
        return (
            f"wss://api.deepgram.com/v1/speak?model={self.model}"
            f"&encoding=linear16&sample_rate={SAMPLE_RATE}&container=none"
        )

    async def start(self) -> None:
        import websockets

        headers = {"Authorization": f"Token {self.api_key}"}
        try:  # websockets >= 13 renamed the kwarg
            self._ws = await websockets.connect(self._url, additional_headers=headers)
        except TypeError:
            self._ws = await websockets.connect(self._url, extra_headers=headers)
        log.info("TTS: deepgram %s connected", self.model)

    async def speak(self, text: str) -> AsyncIterator[bytes]:
        text = _clean(text)
        if not text or self._ws is None:
            return
        try:
            await self._ws.send(json.dumps({"type": "Speak", "text": text}))
            await self._ws.send(json.dumps({"type": "Flush"}))
            while True:
                msg = await self._ws.recv()
                if isinstance(msg, bytes):
                    yield msg
                elif json.loads(msg).get("type") == "Flushed":
                    return
        except asyncio.CancelledError:
            raise  # barge-in; the socket is put back in order by reset()
        except Exception as exc:  # noqa: BLE001 - a dead voice must not kill the call
            log.warning("tts failed for %r: %s", text[:40], exc)

    async def reset(self) -> None:
        """Drop the audio still in flight for an abandoned sentence.

        Called after a barge-in cancels ``speak``. Deepgram keeps sending the
        rest of the sentence until it sees Clear, and acknowledges with
        Cleared - so read past the stale bytes, or the next sentence opens
        with the tail of the one the caller interrupted.
        """
        if self._ws is None:
            return
        deadline = asyncio.get_running_loop().time() + self.DRAIN_TIMEOUT
        try:
            await self._ws.send(json.dumps({"type": "Clear"}))
            while True:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    log.warning("tts clear was not acknowledged in time")
                    return
                msg = await asyncio.wait_for(self._ws.recv(), timeout)
                if not isinstance(msg, bytes) and json.loads(msg).get("type") == "Cleared":
                    return
        except Exception as exc:  # noqa: BLE001 - a stuck voice must not kill the call
            log.warning("tts clear failed: %s", exc)

    async def close(self) -> None:
        if self._ws:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
            self._ws = None


class EdgeTTS:
    """Free, keyless, and only for development - see this module's docstring."""

    name = "edge-tts"

    def __init__(self, voice: str = TTS_VOICE) -> None:
        self.voice = voice

    async def start(self) -> None:
        """Nothing to hold open; edge-tts opens its own session per sentence."""

    async def reset(self) -> None:
        """Cancelling the generator already closed the sentence's stream."""

    async def speak(self, text: str) -> AsyncIterator[bytes]:
        text = _clean(text)
        if not text:
            return
        import edge_tts

        decoder = Mp3ToPcm()
        comm = edge_tts.Communicate(text, self.voice)
        try:
            # aclosing, not a bare `async for`: barge-in cancels this generator
            # mid-sentence and edge-tts must get to close its http session.
            async with aclosing(comm.stream()) as stream:
                async for chunk in stream:
                    if chunk["type"] == "audio" and chunk.get("data"):
                        pcm = decoder.feed(chunk["data"])
                        if pcm:
                            yield pcm
            tail = decoder.flush()
            if tail:
                yield tail
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a dead voice must not kill the call
            log.warning("tts failed for %r: %s", text[:40], exc)

    async def close(self) -> None:
        """Each sentence already closed its own stream."""


def make_tts():
    if DEEPGRAM_API_KEY:
        return DeepgramTTS()
    log.warning(
        "TTS: edge-tts - an undocumented consumer endpoint with no commercial "
        "agreement. Set DEEPGRAM_API_KEY before any real caller hears this."
    )
    return EdgeTTS()


async def prewarm() -> None:
    """First synthesis of the process pays DNS and TLS setup. Spend it at boot
    instead of on the caller's first turn."""
    voice = make_tts()
    try:
        await voice.start()
        async for _ in voice.speak("Ready."):
            break
    except Exception as exc:  # noqa: BLE001
        log.warning("tts prewarm failed: %s", exc)
    finally:
        await voice.close()
