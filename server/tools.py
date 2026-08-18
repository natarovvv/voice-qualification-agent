"""The three callable tools plus their JSON-schema declarations.

Storage is a JSON file per collection - a $0 stand-in for Postgres.
ponytail: file-per-collection behind a process lock; move to Postgres when
more than one worker writes.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from config import DATA_DIR

_LOCK = threading.Lock()
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]{2,}$")
FREE_DOMAINS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "proton.me"}
BUSINESS_START, BUSINESS_END = 9, 17  # UTC hours the calendar accepts
SLOT_MINUTES = 30


def _path(name: str):
    return DATA_DIR / f"{name}.json"


def _load(name: str, default: Any) -> Any:
    try:
        return json.loads(_path(name).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save(name: str, value: Any) -> None:
    _path(name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_company_size(company_size: Any) -> int | None:
    """Accept 50, '50', '50-200', 'about 250 people', 'enterprise'."""
    if isinstance(company_size, (int, float)):
        return int(company_size)
    text = str(company_size or "").lower()
    nums = [int(n) for n in re.findall(r"\d+", text)]
    if nums:
        return max(nums)
    return {"solo": 1, "startup": 10, "smb": 50, "mid-market": 300, "enterprise": 2000}.get(text.strip())


def check_lead_qualification(email: str, company_size: Any) -> dict:
    """Score a lead and persist it. Deterministic, so sales can audit it."""
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email):
        return {"ok": False, "error": "invalid_email", "message": "That email address is not valid."}

    size = parse_company_size(company_size)
    if size is None:
        return {"ok": False, "error": "unknown_company_size", "message": "Ask how many employees they have."}

    domain = email.split("@")[1]
    score = 0
    reasons = []
    if domain in FREE_DOMAINS:
        reasons.append("free email domain")
    else:
        score += 30
        reasons.append("business email domain")
    if size >= 500:
        score += 50
        reasons.append("500+ employees")
    elif size >= 50:
        score += 35
        reasons.append("50-499 employees")
    elif size >= 10:
        score += 15
        reasons.append("10-49 employees")
    else:
        reasons.append("under 10 employees")

    tier = "hot" if score >= 65 else "warm" if score >= 40 else "cold"
    record = {
        "email": email,
        "domain": domain,
        "company_size": size,
        "score": score,
        "tier": tier,
        "reasons": reasons,
        "qualified": tier in ("hot", "warm"),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    with _LOCK:
        leads = _load("leads", [])
        leads.append(record)
        _save("leads", leads)
    return {"ok": True, **record}


def _parse_dt(value: str) -> datetime | None:
    text = str(value or "").strip().replace("Z", "+00:00")
    candidates = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S")
    try:
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in candidates:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def book_calendar_slot(email: str, datetime_iso: str) -> dict:
    """Book a 30-minute slot. Rejects the past, off-hours and double-books."""
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email):
        return {"ok": False, "error": "invalid_email", "message": "That email address is not valid."}

    start = _parse_dt(datetime_iso)
    if start is None:
        return {"ok": False, "error": "invalid_datetime", "message": "Give the time as YYYY-MM-DD HH:MM."}

    now = datetime.now(timezone.utc)
    if start < now:
        return {"ok": False, "error": "in_the_past", "message": "That time has passed. Offer a later one."}
    if start.weekday() >= 5 or not (BUSINESS_START <= start.hour < BUSINESS_END):
        return {
            "ok": False,
            "error": "outside_business_hours",
            "message": "Slots run weekdays 09:00-17:00 UTC. Offer the nearest one.",
        }

    end = start + timedelta(minutes=SLOT_MINUTES)
    with _LOCK:
        bookings = _load("bookings", [])
        for b in bookings:
            b_start = _parse_dt(b["start"])
            if b_start and b_start < end and start < b_start + timedelta(minutes=SLOT_MINUTES):
                return {
                    "ok": False,
                    "error": "slot_taken",
                    "message": f"{start:%A %H:%M} UTC is taken. Suggest {end:%H:%M} UTC.",
                }
        record = {
            "email": email,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "booked_at": now.isoformat(),
        }
        bookings.append(record)
        _save("bookings", bookings)
    return {"ok": True, "confirmation": f"{start:%A %d %B at %H:%M} UTC", **record}


# Words that match everything and therefore mean nothing. Without these,
# "do you sell bicycles" scores a hit on every article containing "you".
STOPWORDS = frozenset(
    "the and are was were you your our its for with that this have has had can "
    "could would should does did what when where which who how why not but any "
    "about from they them there here get got let want need know tell say".split()
)
MIN_RELEVANCE = 0.34  # at least a third of the meaningful words must land


def _terms(text: str, stop: bool = True) -> set[str]:
    words = {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}
    return words - STOPWORDS if stop else words


def lookup_kb(query: str) -> dict:
    """Whole-word overlap search over the KB.

    ponytail: term overlap, not embeddings - swap in a vector store when the
    KB outgrows a few dozen articles.
    """
    query = str(query or "").strip()
    if not query:
        return {"ok": False, "error": "empty_query", "message": "Ask the caller what they need to know."}

    terms = _terms(query)
    if not terms:
        return {"ok": True, "found": False, "message": "Nothing to search on. Ask them to say more."}

    scored = []
    for art in _load("kb", []):
        labels = _terms(f"{art['title']} {' '.join(art.get('tags', []))}", stop=False)
        words = labels | _terms(art["body"], stop=False)
        relevance = len(terms & words) / len(terms)
        if relevance >= MIN_RELEVANCE:
            # a hit in the title or tags beats the same hit buried in prose:
            # "cost" matches Pricing's tags but only Onboarding's small print
            scored.append((relevance, len(terms & labels), art))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)

    if not scored:
        return {"ok": True, "found": False, "message": "Nothing in the knowledge base covers that. Do not guess."}
    return {
        "ok": True,
        "found": True,
        "results": [{"title": a["title"], "body": a["body"], "relevance": round(s, 2)} for s, _, a in scored[:3]],
    }


SCHEMAS: list[dict] = [
    {
        "name": "check_lead_qualification",
        "description": (
            "Score an inbound lead once you know their email and company size. "
            "Call it before offering a meeting."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "The caller's work email address."},
                "company_size": {"type": "string", "description": "Employee count, e.g. '250' or '50-200'."},
            },
            "required": ["email", "company_size"],
        },
    },
    {
        "name": "book_calendar_slot",
        "description": "Book a 30-minute specialist call. Weekdays 09:00-17:00 UTC only.",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "The caller's email address."},
                "datetime_iso": {"type": "string", "description": "Slot start in UTC as YYYY-MM-DD HH:MM."},
            },
            "required": ["email", "datetime_iso"],
        },
    },
    {
        "name": "lookup_kb",
        "description": "Search the product knowledge base. Use it for any product, pricing or policy question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What the caller asked, in a few words."},
            },
            "required": ["query"],
        },
    },
]

REGISTRY: dict[str, Callable[..., dict]] = {
    "check_lead_qualification": check_lead_qualification,
    "book_calendar_slot": book_calendar_slot,
    "lookup_kb": lookup_kb,
}


def call(name: str, args: dict) -> dict:
    """Dispatch a model-requested tool call. Never raises - the model reads the error."""
    fn = REGISTRY.get(name)
    if fn is None:
        return {"ok": False, "error": "unknown_tool", "message": f"No tool named {name}."}
    allowed = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    try:
        return fn(**{k: v for k, v in (args or {}).items() if k in allowed})
    except TypeError as exc:
        return {"ok": False, "error": "bad_arguments", "message": str(exc)}
