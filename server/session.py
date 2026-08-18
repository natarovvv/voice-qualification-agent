"""In-memory session store with TTL, plus the end-of-call record writer.

PRD allows "Redis / in-memory dict with TTL". One process, one dict.
ponytail: swap ``SessionStore`` for Redis when you run more than one worker.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from config import DATA_DIR, MAX_HISTORY_TURNS, MAX_TURN_CHARS, SESSION_TTL

CALLS_DIR = DATA_DIR / "calls"
CALLS_DIR.mkdir(parents=True, exist_ok=True)

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


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
        (CALLS_DIR / f"{self.id}.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.ended = True
        return rec


class SessionStore:
    def __init__(self, ttl: int = SESSION_TTL) -> None:
        self.ttl = ttl
        self._items: dict[str, Session] = {}

    def get(self, session_id: str | None) -> Session:
        self.sweep()
        sid = session_id or uuid.uuid4().hex[:12]
        s = self._items.get(sid)
        if s is None or s.ended:
            s = Session(id=sid)
            self._items[sid] = s
        s.touch()
        return s

    def sweep(self, now: float | None = None) -> int:
        now = now or time.time()
        dead = [k for k, v in self._items.items() if now - v.touched_at > self.ttl]
        for k in dead:
            del self._items[k]
        return len(dead)

    def __len__(self) -> int:
        return len(self._items)


STORE = SessionStore()
