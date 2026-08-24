"""A worker process for the two-worker test: the real app, minus the voice.

The only thing swapped out is TTS. With no DEEPGRAM_API_KEY the server falls
back to edge-tts, and edge-tts is a network call to an endpoint the README
already calls development-only - not something a test should hang off. What
this test is actually about, the shared session store and the drop timer and
the hold between them, is the shipped code running in a shipped process.

Run as: python mute_worker.py <port>
"""
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import tts  # noqa: E402


class MuteTTS:
    name = "mute"

    async def start(self):
        pass

    async def speak(self, text: str):
        await asyncio.sleep(0)
        return
        yield b""  # never reached; this is what makes speak an async generator

    async def reset(self):
        pass

    async def close(self):
        pass


tts.make_tts = lambda: MuteTTS()
tts.prewarm = lambda: asyncio.sleep(0)

import main  # noqa: E402
import uvicorn  # noqa: E402

uvicorn.run(main.app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
