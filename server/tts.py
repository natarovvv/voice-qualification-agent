"""Text to speech: edge-tts (free, no key) -> PCM 16 kHz 16-bit mono.

edge-tts hands back 24 kHz MP3, so we decode and resample as the chunks
arrive - buffering the whole clip first would cost us the latency budget.
PyAV ships its own ffmpeg binaries, so there is no system dependency.
"""
from __future__ import annotations

import asyncio
from contextlib import aclosing
import logging
import re
from typing import AsyncIterator

from config import SAMPLE_RATE, TTS_VOICE

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


async def speak(text: str, voice: str = TTS_VOICE) -> AsyncIterator[bytes]:
    """Yield PCM chunks for one sentence. Cancel the task to stop mid-word."""
    text = _clean(text)
    if not text:
        return
    import edge_tts

    decoder = Mp3ToPcm()
    comm = edge_tts.Communicate(text, voice)
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


async def prewarm(voice: str = TTS_VOICE) -> None:
    """First synthesis of the process pays ~900 ms of DNS and TLS setup.
    Spend it at boot instead of on the caller's first turn."""
    try:
        async for _ in speak("Ready.", voice):
            break
    except Exception as exc:  # noqa: BLE001
        log.warning("tts prewarm failed: %s", exc)
