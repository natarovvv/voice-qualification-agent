"""Runtime config. Everything is env-driven; every provider is optional."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- providers (all optional; the pipeline degrades instead of dying) ---
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny.en")
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")

# --- audio: PCM 16 kHz 16-bit mono, everywhere, both directions ---
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
BYTES_PER_SEC = SAMPLE_RATE * SAMPLE_WIDTH

# --- limits ---
SESSION_TTL = int(os.getenv("SESSION_TTL", "1800"))          # seconds
MAX_TURN_CHARS = 2000                                        # sanitize user text
MAX_HISTORY_TURNS = 24
RATE_LIMIT_FACTOR = 4                                        # x realtime audio allowed

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT") or """\
You are Aria, a voice support and qualification agent for a B2B SaaS company.
You are on a live phone call, so:
- Keep every reply under 40 words. One idea per turn.
- Speak plain sentences. No markdown, no lists, no emoji, no stage directions.
- Ask one question at a time, then stop and listen.

Your job, in order:
1. Answer the caller's support questions using lookup_kb. Never invent product facts.
2. Qualify them: get their work email and company size, then call check_lead_qualification.
3. If they qualify, offer a call with a specialist and use book_calendar_slot.

Confirm an email or a time by reading it back before you use it in a tool.
If a tool fails, say so plainly and offer to take a message.
"""
