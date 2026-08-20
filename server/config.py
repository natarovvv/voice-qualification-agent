"""Runtime config. Everything is env-driven; every provider is optional."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- providers (all optional; the pipeline degrades instead of dying) ---
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")                      # "groq" | "gemini"; blank = fastest first
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")  # 2.0-flash is retired
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")        # llama-3.3-70b is gone
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny.en")
DEEPGRAM_TTS_MODEL = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")  # voice = model here
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")                    # edge-tts fallback only

# --- audio: PCM 16 kHz 16-bit mono, everywhere, both directions ---
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
BYTES_PER_SEC = SAMPLE_RATE * SAMPLE_WIDTH

# --- echo guard: the agent's own voice coming back through the speakers ---
# Room-dependent, so they are knobs. ECHO_THRESHOLD at 0.5 disables the duck
# (right for a headset); raise it in a room with loud speakers.
ECHO_THRESHOLD = float(os.getenv("ECHO_THRESHOLD", "0.85"))
ECHO_START_MS = int(os.getenv("ECHO_START_MS", "400"))
ECHO_TAIL = int(os.getenv("ECHO_TAIL_MS", "250")) / 1000  # speaker + jitter buffer

# --- who may open a call ---
# The websocket has no same-origin policy of its own, so an allowlist here is
# the only thing stopping any page on the internet from spending your STT and
# LLM budget. "*" turns the check off; do not ship that.
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()
]
# Shared secret for /ws. Empty = open, which is only safe while HOST is
# loopback. A browser build has to ship this to the client, so it stops
# scanners and other people's pages, not a determined caller - real per-user
# auth needs a login the PRD does not have.
# ponytail: shared secret; swap for a signed short-lived ticket when there are accounts.
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# --- limits ---
SESSION_TTL = int(os.getenv("SESSION_TTL", "1800"))          # seconds
# How long a call whose socket died stays resumable before its record is
# written. A dropped connection is not a hangup - a tunnel dies, a phone
# changes network, a tab reloads - and ending the call there throws away
# everything said so far. 0 disables it: every disconnect ends the call.
RESUME_GRACE = float(os.getenv("RESUME_GRACE", "60"))        # seconds
# Shared session store. Unset = a process-local dict, which is what one worker
# wants anyway. See the note in session.py about what this does and does not fix.
REDIS_URL = os.getenv("REDIS_URL", "")
# Leads and bookings. Unset keeps the JSON files, which need no setup and are
# single-process only. Postgres is what makes a second worker safe: the "is
# this slot free" check becomes an EXCLUDE constraint instead of a lock.
DATABASE_URL = os.getenv("DATABASE_URL", "")
MAX_TURN_CHARS = 2000                                        # sanitize user text
MAX_HISTORY_TURNS = 24
RATE_LIMIT_FACTOR = 4                                        # x realtime audio allowed
MAX_CALLS = int(os.getenv("MAX_CALLS", "20"))                # concurrent websockets
MAX_TEXT_TURNS = 10                                          # typed turns per window
TEXT_WINDOW = 10.0                                           # seconds
CALL_RETENTION_DAYS = int(os.getenv("CALL_RETENTION_DAYS", "30"))  # 0 = keep forever

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
