"""Session store with TTL, plus the end-of-call record writer.

PRD allows "Redis / in-memory dict with TTL", and both are here: set REDIS_URL
for the shared one, leave it unset for a process-local dict. Same async
interface either way, so the call path does not know which it got.

A session outlives its websocket. A dropped socket is not a hangup, so the call
stays here for RESUME_GRACE and a reconnect carrying its id picks up the same
conversation; only a caller who actually hangs up (or a grace period nobody
came back for) writes the record and ends the session. That is what the store
is for, and with REDIS_URL the reconnect survives the worker restarting too.

What Redis does not buy is a second worker. A call still lives on one worker
for its whole life, and the drop timer is that worker's own. The thing that
actually blocks a second worker is storage.py's JSON backend: leads and
bookings behind a threading.Lock, which is process-local. DATABASE_URL is the
setting that fixes that one.
"""
from __future__ import annotations

import json
import logging
import re
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from config import (
    CALL_ENCRYPTION_KEYS,
    CALL_RETENTION_DAYS,
    DATA_DIR,
    MAX_HISTORY_TURNS,
    MAX_TURN_CHARS,
    REDIS_URL,
    SESSION_TTL,
)

log = logging.getLogger(__name__)

CALLS_DIR = DATA_DIR / "calls"
CALLS_DIR.mkdir(parents=True, exist_ok=True)

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# A session id becomes a filename, so it may never carry a separator, a dot or
# a drive letter. Checked once, at construction, so every path built from an
# id downstream is safe by construction rather than by remembering to check.
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


class Unreadable(Exception):
    """A record on disk this process cannot open: truncated, or sealed with a
    key this deployment does not have. Either way its contents are unknown,
    which matters most to erasure - an unreadable record is one not erased."""


def _make_cipher():
    """The cipher for call records, or None when no key is configured.

    Fernet: AES-CBC with an HMAC over the ciphertext, from a library that is
    audited. Rolling our own is the one thing here that would be indefensible.
    MultiFernet takes the list so a key can be rotated without a migration -
    the first key seals, any of them opens.
    """
    if not CALL_ENCRYPTION_KEYS:
        return None
    try:
        from cryptography.fernet import Fernet, MultiFernet
    except ImportError as exc:  # refuse to start rather than quietly not encrypt
        raise RuntimeError(
            "CALL_ENCRYPTION_KEY is set but cryptography is not installed. "
            "Writing plaintext you believe is encrypted is worse than not starting."
        ) from exc
    return MultiFernet([Fernet(k) for k in CALL_ENCRYPTION_KEYS])


CIPHER = _make_cipher()


def seal(payload: Any, keep: dict | None = None) -> Any:
    """Anything JSON-serialisable, on its way to disk.

    ``keep`` is what stays legible beside the ciphertext. A call record keeps
    its session id: that is the filename already, retention has to find a file
    without opening it, and it is a random token rather than anything about
    the caller. Leads and bookings keep nothing - the rows are the payload.
    """
    if CIPHER is None:
        return payload
    token = CIPHER.encrypt(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return {**(keep or {}), "enc": "fernet", "data": token.decode()}


def unseal(blob: Any, where: str = "record") -> Any:
    """The inverse, for a value already loaded off disk.

    Anything without the wrapper is returned as it came: written before a key
    was configured, and still readable. That is what makes turning encryption
    on not a migration - sealed and plain files live side by side.
    """
    if not (isinstance(blob, dict) and blob.get("enc")):
        return blob
    if CIPHER is None:
        raise Unreadable(f"{where}: sealed, and no CALL_ENCRYPTION_KEY is set")
    try:
        return json.loads(CIPHER.decrypt(blob["data"].encode()))
    except Exception as exc:  # noqa: BLE001 - InvalidToken, and anything else
        raise Unreadable(f"{where}: {type(exc).__name__}") from exc


def read_record(path) -> dict:
    """A call record on its way back, sealed or not."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Unreadable(f"{path.name}: {exc}") from exc
    return unseal(blob, path.name)


def purge_old_calls(days: int = CALL_RETENTION_DAYS) -> int:
    """Drop call records past the retention window. Transcripts are personal
    data; keeping them forever is a liability, not a feature."""
    if days <= 0:
        return 0
    cutoff = time.time() - days * 86400
    gone = 0
    for f in CALLS_DIR.glob("*.json"):
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
            gone += 1
    return gone


def write_json(path, value) -> None:
    """Write to a temp file in the same directory, then rename over the target.

    A half-written JSON file reads back as a parse error, and every loader here
    treats a parse error as "empty" - so a crash mid-write would silently
    discard the whole collection instead of losing the last record.
    """
    tmp = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def sanitize(text: str) -> str:
    """Trust boundary: caller speech goes straight into a prompt.

    Strip control characters, collapse whitespace, cap length, and neutralise
    the role markers a caller could read aloud to fake a system turn.
    """
    text = _CONTROL.sub(" ", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()[:MAX_TURN_CHARS]
    return re.sub(
        r"(?i)\b(system|assistant|user)\s*:",
        lambda m: m.group(0).replace(":", " -"),
        text,
    )


@dataclass
class Session:
    id: str
    created_at: float = field(default_factory=time.time)
    touched_at: float = field(default_factory=time.time)
    history: list[dict] = field(default_factory=list)   # {"role","content"} for the LLM
    transcript: list[dict] = field(default_factory=list)  # {"at","speaker","text"}
    tool_calls: list[dict] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)  # email, company_size, tier, booking
    ended: bool = False

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.id):
            raise ValueError(f"unsafe session id: {self.id!r}")

    def touch(self) -> None:
        self.touched_at = time.time()

    def add_turn(self, role: str, content: str) -> None:
        content = sanitize(content) if role == "user" else str(content or "").strip()
        if not content:
            return
        self.history.append({"role": role, "content": content})
        # keep the tail; the system prompt lives outside history
        if len(self.history) > MAX_HISTORY_TURNS:
            del self.history[: len(self.history) - MAX_HISTORY_TURNS]
        self.transcript.append(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "speaker": "caller" if role == "user" else "agent",
                "text": content,
            }
        )
        self.touch()

    def add_tool_call(self, name: str, args: dict, result: dict) -> None:
        self.tool_calls.append(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "name": name,
                "args": args,
                "result": result,
            }
        )
        if name == "check_lead_qualification" and result.get("ok"):
            self.facts.update(
                email=result["email"],
                company_size=result["company_size"],
                tier=result["tier"],
                score=result["score"],
                qualified=result["qualified"],
            )
        elif name == "book_calendar_slot" and result.get("ok"):
            self.facts["booking"] = {"start": result["start"], "email": result["email"]}
        self.touch()

    def record(self, summary: dict | None = None) -> dict:
        return {
            "session_id": self.id,
            "started_at": datetime.fromtimestamp(self.created_at, timezone.utc).isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "duration_sec": round(time.time() - self.created_at, 1),
            "lead": self.facts,
            "summary": summary or {},
            "tool_calls": self.tool_calls,
            "transcript": self.transcript,
        }

    def save(self, summary: dict | None = None) -> dict:
        """Structured output at call end - one JSON file per call."""
        rec = self.record(summary)
        write_json(CALLS_DIR / f"{self.id}.json", seal(rec, keep={"session_id": self.id}))
        self.ended = True
        return rec  # the caller gets their own call back; only the disk copy is sealed


def _new_session() -> Session:
    """The caller never chooses an id. An id it can choose is an id it can
    guess, and guessing one lends it someone else's transcript and lead data."""
    return Session(id=secrets.token_urlsafe(18))


def _evict(items: dict[str, Session], ttl: int, now: float | None = None) -> int:
    now = now or time.time()
    dead = [k for k, v in items.items() if now - v.touched_at > ttl]
    for k in dead:
        del items[k]
    return len(dead)


class SessionStore:
    """One process, one dict. The default."""

    name = "memory"

    def __init__(self, ttl: int = SESSION_TTL) -> None:
        self.ttl = ttl
        self._items: dict[str, Session] = {}

    async def get(self, session_id: str | None) -> Session:
        """Resume a live session, or mint a new one."""
        self.sweep()
        s = self._items.get(session_id) if session_id else None
        if s is None or s.ended:
            s = _new_session()
            self._items[s.id] = s
        s.touch()
        return s

    async def put(self, session: Session) -> None:
        """Already by reference; nothing to write."""

    async def drop(self, session_id: str) -> None:
        self._items.pop(session_id, None)

    def sweep(self, now: float | None = None) -> int:
        return _evict(self._items, self.ttl, now)

    def __len__(self) -> int:
        return len(self._items)


class RedisSessionStore:
    """The same store, shared, so an interrupted call is recoverable.

    Sessions the worker is currently serving stay in a local dict as well: that
    copy is the newest one while its websocket is open, and it means a Redis
    outage degrades to exactly the in-memory behaviour instead of dropping the
    caller's history mid-sentence. Redis expiry replaces the manual sweep for
    the shared copy; the local one is still swept, or an abandoned call would
    sit in it forever.
    """

    name = "redis"

    def __init__(self, url: str = REDIS_URL, ttl: int = SESSION_TTL) -> None:
        import redis.asyncio

        self.ttl = ttl
        # from_url does not connect here - the pool dials on the first command,
        # so a Redis that is down cannot stop the process from starting.
        self.redis = redis.asyncio.from_url(url, decode_responses=True)
        self._live: dict[str, Session] = {}

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}"

    async def get(self, session_id: str | None) -> Session:
        _evict(self._live, self.ttl)
        s = None
        if session_id:
            s = self._live.get(session_id) or await self._load(session_id)
        if s is None or s.ended:
            s = _new_session()
        self._live[s.id] = s
        s.touch()
        return s

    async def _load(self, session_id: str) -> Session | None:
        try:
            raw = await self.redis.get(self._key(session_id))
        except Exception as exc:  # noqa: BLE001 - a dead cache must not refuse the call
            log.warning("redis read failed; starting a fresh session: %s", exc)
            return None
        if not raw:
            return None
        try:
            # Session validates its own id, so a hostile or corrupted value
            # cannot come back as something that escapes the calls directory.
            return Session(**json.loads(raw))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("unreadable session %s in redis: %s", session_id[:8], exc)
            return None

    async def put(self, session: Session) -> None:
        self._live[session.id] = session
        try:
            await self.redis.set(
                self._key(session.id), json.dumps(asdict(session)), ex=self.ttl
            )
        except Exception as exc:  # noqa: BLE001 - the call continues from memory
            log.warning("redis write failed: %s", exc)

    async def drop(self, session_id: str) -> None:
        self._live.pop(session_id, None)
        try:
            await self.redis.delete(self._key(session_id))
        except Exception as exc:  # noqa: BLE001
            log.warning("redis delete failed: %s", exc)

    def __len__(self) -> int:
        return len(self._live)


def make_store():
    if REDIS_URL:
        try:
            return RedisSessionStore()
        except ImportError:
            log.warning("REDIS_URL is set but redis is not installed; using memory")
    return SessionStore()


STORE = make_store()
