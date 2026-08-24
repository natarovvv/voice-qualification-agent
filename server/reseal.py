"""Re-seal the rows already in Postgres, under whatever key is configured now.

A file re-seals itself, because a file is rewritten whole the next time a row
is added to it. A row is not. A lead written before CALL_ENCRYPTION_KEY was
set keeps its address in the clear for as long as nobody touches it, and one
written under a key that has since been rotated out keeps that key's hash.

Neither is broken: a lookup goes out under every key with the plain address
beside them, so erasure still reaches both. That is what makes turning the key
on not a migration. It is also why "the key is on" and "the data is sealed"
are two different sentences until this has been run once.

    python reseal.py              # do it
    python reseal.py --dry-run    # count what it would touch, change nothing

Idempotent, so the second run reports nothing to do: a row is rewritten only
when its lookup is not already the one the current first key would write.
"""
from __future__ import annotations

import logging
import sys

import session
from config import DATABASE_URL
from session import Unreadable, blind, seal, unseal

log = logging.getLogger(__name__)

TABLES = ("leads", "bookings")


def reseal(store, dry_run: bool = False) -> dict:
    """Rewrite every row that is not already sealed under the current key."""
    if not session.CALL_ENCRYPTION_KEYS:
        raise SystemExit(
            "CALL_ENCRYPTION_KEY is not set, so there is nothing to re-seal under. "
            "Set it first; this script does not remove a seal."
        )
    from psycopg.types.json import Json

    store.ensure_schema()
    counts = {}
    with store.pool.connection() as conn:
        for table in TABLES:
            done = already = unreadable = 0
            # ponytail: the whole table in memory. These are lead rows, not a
            # ledger; page by id if one ever gets big enough to notice.
            rows = conn.execute(f"SELECT id, email, contact FROM {table} ORDER BY id").fetchall()
            for row_id, stored, contact in rows:
                try:
                    # The address is in contact once there is a seal, and in
                    # the email column before there is one. Reading it back is
                    # the only reason the sealed copy exists.
                    sealed = unseal(contact, f"{table}.{row_id}") if contact else None
                except Unreadable as exc:
                    # A row under a key this deployment does not have. Counted
                    # rather than skipped quietly: same reasoning as erasure -
                    # a row nobody can read is a row nobody can prove is sealed.
                    log.warning("cannot re-seal %s", exc)
                    unreadable += 1
                    continue
                address = sealed["email"] if sealed else stored
                # Written under the first key, found under any of them - the
                # same asymmetry the seal has, and the reason this is a
                # retrofit rather than something that has to run before the
                # key does.
                lookup = blind(address)[0]
                if stored == lookup:
                    already += 1
                    continue
                if sealed is None:
                    # A row from before the key. The domain that used to be a
                    # column of its own is the address with the name cut off,
                    # so it comes back without having been kept anywhere.
                    sealed = {"email": address}
                    if table == "leads":
                        sealed["domain"] = address.split("@")[-1]
                if not dry_run:
                    conn.execute(
                        f"UPDATE {table} SET email = %s, contact = %s WHERE id = %s",
                        (lookup, Json(seal(sealed)), row_id),
                    )
                done += 1
            counts[table] = {"resealed": done, "already": already, "unreadable": unreadable}
    return counts


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dry_run = "--dry-run" in argv
    if not DATABASE_URL:
        print("DATABASE_URL is not set: leads and bookings are files, and a file "
              "re-seals itself the next time a row is added to it.")
        return 0

    from storage import PostgresStore

    store = PostgresStore()
    try:
        counts = reseal(store, dry_run=dry_run)
    finally:
        store.pool.close()
    for table, c in counts.items():
        print(f"{table}: {c['resealed']} {'would be ' if dry_run else ''}re-sealed, "
              f"{c['already']} already current, {c['unreadable']} unreadable")
    return 1 if any(c["unreadable"] for c in counts.values()) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
