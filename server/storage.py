"""Where leads and bookings live.

Two backends behind one interface, chosen by DATABASE_URL - JSON files when it
is unset, Postgres when it is set. Same reasoning as REDIS_URL: the demo has to
keep running with nothing configured.

The reason to bother is the calendar. On files, "is this slot free" is a read,
a check and an append under a threading.Lock, and that lock means nothing
across processes: two workers will happily sell the same slot twice. On
Postgres the overlap is an EXCLUDE constraint, so the database refuses the
second booking no matter how many workers ask at once. That is the invariant
moving out of application code and into the one place that can actually hold
it.

Both backends seal what they hold when CALL_ENCRYPTION_KEY is set. A file is
sealed whole. A row cannot be, because the index and the overlap constraint
have to keep working on it, so the address becomes an HMAC the index can still
match - see session.blind - and the address itself goes sealed into a column
of its own. Nothing ever reads it back here; equality is all these queries
ever asked of it.

The knowledge base is not here. It is content rather than caller data, and it
stays a file that ships with the repo.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from typing import Any

from config import DATABASE_URL, DATA_DIR
from session import Unreadable, blind, seal, unseal, write_json

log = logging.getLogger(__name__)

MAX_RECORDS = 5000  # json backend only; the file is read whole on every call

SCHEMA = """
CREATE TABLE IF NOT EXISTS leads (
    id           bigserial PRIMARY KEY,
    -- Not the address. session.blind's HMAC of it when a key is set, the
    -- address itself when not: equality and an index still work, reading it
    -- back does not, and that is the whole trade.
    email        text        NOT NULL,
    -- The address itself, sealed. Also where the domain went - it is
    -- email.split("@")[1] and nothing more, so a column of its own was a
    -- second copy of the same personal datum sitting in the clear.
    contact      jsonb,
    company_size integer     NOT NULL,
    score        integer     NOT NULL,
    tier         text        NOT NULL,
    reasons      jsonb       NOT NULL DEFAULT '[]'::jsonb,
    qualified    boolean     NOT NULL,
    checked_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS leads_email_idx ON leads (email);

CREATE TABLE IF NOT EXISTS bookings (
    id        bigserial PRIMARY KEY,
    email     text        NOT NULL,  -- the same lookup value, for the same reason
    contact   jsonb,
    -- These two stay in the clear because the constraint below is the reason
    -- this table exists, and a range index cannot read ciphertext. A slot is
    -- not personal data on its own; who is in it is, and that is sealed.
    start_at  timestamptz NOT NULL,
    end_at    timestamptz NOT NULL,
    booked_at timestamptz NOT NULL DEFAULT now(),
    -- The whole reason this table is not a JSON file: two workers cannot both
    -- be told the slot was free. A plain gist index is enough here because the
    -- constraint is a range overlap alone, with no scalar equality mixed in,
    -- so btree_gist is not needed.
    CONSTRAINT bookings_no_overlap
        EXCLUDE USING gist (tstzrange(start_at, end_at) WITH &&)
);
CREATE INDEX IF NOT EXISTS bookings_email_idx ON bookings (email);

-- Tables that predate all of the above. Dropping domain loses nothing: those
-- rows still hold their address in the clear, and the domain is a substring
-- of it.
ALTER TABLE leads    ADD COLUMN IF NOT EXISTS contact jsonb;
ALTER TABLE leads    DROP COLUMN IF EXISTS domain;
ALTER TABLE bookings ADD COLUMN IF NOT EXISTS contact jsonb;
"""


class JsonStore:
    """A file per collection. No setup, one process, and it says so."""

    name = "json"

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def _path(self, name: str):
        return DATA_DIR / f"{name}.json"

    def _load(self, name: str) -> list:
        try:
            blob = json.loads(self._path(name).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        # Deliberately not caught: a file sealed with a key this process does
        # not have must not read back as "no rows", or the next add_lead would
        # write a fresh list over the top of it. tools.call turns this into a
        # tool failure the agent can talk about, and nothing is written.
        return unseal(blob, f"{name}.json")

    def _save(self, name: str, rows: list) -> None:
        # Nothing is kept legible here: unlike a call record, the file is found
        # by its own name and nothing needs to identify a row without the key.
        write_json(self._path(name), seal(rows[-MAX_RECORDS:]))

    def add_lead(self, record: dict) -> None:
        with self._lock:
            rows = self._load("leads")
            rows.append(record)
            self._save("leads", rows)

    def book(self, record: dict, slot_minutes: int, cap: int) -> str | None:
        """Insert unless it overlaps. Returns None on success, else a reason."""
        from datetime import timedelta

        start = datetime.fromisoformat(record["start"])
        end = start + timedelta(minutes=slot_minutes)
        with self._lock:
            rows = self._load("bookings")
            if sum(1 for b in rows if b.get("email") == record["email"]) >= cap:
                return "too_many_bookings"
            for b in rows:
                b_start = datetime.fromisoformat(b["start"])
                if b_start < end and start < b_start + timedelta(minutes=slot_minutes):
                    return "slot_taken"
            rows.append(record)
            self._save("bookings", rows)
        return None

    def erase(self, email: str) -> dict:
        removed = {}
        with self._lock:
            for name in ("leads", "bookings"):
                rows = self._load(name)
                kept = [r for r in rows if str(r.get("email", "")).lower() != email]
                removed[name] = len(rows) - len(kept)
                if removed[name]:
                    self._save(name, kept)
        return removed


class PostgresStore:
    """The same two collections, where a second worker cannot corrupt them."""

    name = "postgres"

    def __init__(self, url: str = DATABASE_URL) -> None:
        from psycopg_pool import ConnectionPool

        # open=False so an unreachable database cannot stop the process from
        # starting; the first tool call opens it and reports its own failure.
        self.pool = ConnectionPool(url, min_size=1, max_size=4, open=False)
        self._ready = False

    def ensure_schema(self) -> None:
        if self._ready:
            return
        self.pool.open()
        with self.pool.connection() as conn:
            conn.execute(SCHEMA)
        self._ready = True
        log.info("postgres: leads and bookings ready")

    def add_lead(self, record: dict) -> None:
        from psycopg.types.json import Json

        self.ensure_schema()
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO leads (email, contact, company_size, score, tier, reasons,"
                " qualified, checked_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    blind(record["email"])[0],
                    Json(seal({"email": record["email"], "domain": record["domain"]})),
                    record["company_size"], record["score"], record["tier"],
                    Json(record["reasons"]), record["qualified"], record["checked_at"],
                ),
            )

    def book(self, record: dict, slot_minutes: int, cap: int) -> str | None:
        import psycopg
        from psycopg.types.json import Json

        self.ensure_schema()
        lookups = blind(record["email"])
        try:
            with self.pool.connection() as conn, conn.transaction():
                # ponytail: read-committed, so two simultaneous bookings can
                # both pass the cap and land at cap+1. It is a spam guard, not
                # money; the overlap below is the invariant that has to hold,
                # and that one the constraint enforces whatever happens here.
                taken = conn.execute(
                    "SELECT count(*) FROM bookings WHERE email = ANY(%s)", (lookups,)
                ).fetchone()[0]
                if taken >= cap:
                    return "too_many_bookings"
                conn.execute(
                    "INSERT INTO bookings (email, contact, start_at, end_at, booked_at)"
                    " VALUES (%s,%s,%s,%s,%s)",
                    (lookups[0], Json(seal({"email": record["email"]})),
                     record["start"], record["end"], record["booked_at"]),
                )
        except psycopg.errors.ExclusionViolation:
            # Somebody else holds that slot. The database decided this, so it
            # is true even when the deciding worker is not this one.
            return "slot_taken"
        return None

    def erase(self, email: str) -> dict:
        self.ensure_schema()
        # The plain address goes in the list beside the hashes: rows written
        # before a key was configured still hold it, and a row erasure cannot
        # see is a row that does not get erased. Same reasoning as unseal
        # returning an unwrapped payload as it came.
        lookups = blind(email) + [(email or "").strip().lower()]
        removed = {}
        with self.pool.connection() as conn:
            for table in ("leads", "bookings"):
                cur = conn.execute(
                    f"DELETE FROM {table} WHERE lower(email) = ANY(%s)", (lookups,)
                )
                removed[table] = cur.rowcount
        return removed


def make_storage() -> Any:
    if DATABASE_URL:
        try:
            return PostgresStore()
        except ImportError:
            log.warning("DATABASE_URL is set but psycopg is not installed; using files")
    return JsonStore()


STORAGE = make_storage()
