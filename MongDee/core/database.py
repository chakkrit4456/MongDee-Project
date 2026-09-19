"""SQLite storage for MONGDEE AI Booth OS.

Each call opens a short-lived connection — write volume from a single booth
is low (one row per recognized product / health transition / heartbeat), so
this keeps the module trivially thread-safe without a shared connection.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "mongdee.db"


def _assert_not_production_under_test(db_path: Path) -> None:
    """Refuse to touch the real production DB while pytest is running.

    Past incident: an orphaned background test process wrote straight to
    data/mongdee.db and permanently destroyed a real booth's rows (no prior
    commit had the booths table yet, so it was unrecoverable). PYTEST_CURRENT_TEST
    is set by pytest for the whole run, so this check is a real guard, not
    just fixture discipline that a future test can forget to follow.
    """
    if "PYTEST_CURRENT_TEST" not in os.environ:
        return
    try:
        resolved = Path(db_path).resolve()
    except OSError:
        return
    if resolved == DEFAULT_DB_PATH.resolve():
        raise RuntimeError(
            "REFUSING TO RUN TESTS AGAINST PRODUCTION DATABASE: "
            f"{resolved} — pass an isolated --db path (e.g. tests/.data/*.db) instead."
        )

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    booth_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    camera_id TEXT,
    product_key TEXT NOT NULL,
    product_name TEXT NOT NULL,
    confidence REAL,
    question TEXT,
    answer TEXT
);

CREATE TABLE IF NOT EXISTS health_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    booth_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    camera_id TEXT,
    component TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT
);

CREATE TABLE IF NOT EXISTS readiness_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    booth_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    overall_status TEXT NOT NULL,
    detail_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS heartbeats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    booth_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    status TEXT NOT NULL,
    active_cameras INTEGER
);

-- global_person_id/gender/age_category (+ their confidences) below are a
-- read-only snapshot of whatever core.reid.GlobalIdentityRegistry +
-- core.attributes.GlobalPersonAttributeSmoother already knew about the
-- holder at the moment this hold ended (spec: MongDee person-gender/age
-- master prompt sections 22/26/27 "Gender/Age Attribution") — never a fresh
-- inference, and never rewritten after the fact (an Interest Event's
-- attribution must not silently change once a Global Person's profile is
-- later refined by more samples). All five are nullable: a hold logged
-- before this existed, or with Re-ID/FairFace not configured, or where
-- Re-ID simply hadn't resolved that track yet, honestly has no attribution
-- data rather than a fabricated guess.
CREATE TABLE IF NOT EXISTS product_hold_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    booth_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    product_key TEXT NOT NULL,
    product_name TEXT NOT NULL,
    holder_track_id INTEGER NOT NULL,
    hold_start_ts REAL NOT NULL,
    hold_end_ts REAL NOT NULL,
    duration_sec REAL NOT NULL,
    global_person_id TEXT,
    gender TEXT,
    gender_confidence REAL,
    age_category TEXT,
    age_confidence REAL,
    -- 'confirmed' (open_product_interest_event: hold_end_ts/duration_sec
    -- are still provisional) or 'finalized' (finalize_product_interest_
    -- event has since overwritten them with the true values) — spec
    -- section 20/23 "Real-Time 2-Second Confirmation". Existing rows from
    -- before this existed (always inserted with real end/duration already
    -- known) default to 'finalized', which is accurate for them.
    status TEXT NOT NULL DEFAULT 'finalized',
    confirmed_at REAL
);

CREATE TABLE IF NOT EXISTS presence_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    booth_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    duration_sec REAL NOT NULL,
    category TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS booths (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    event_id TEXT,
    created_ts REAL NOT NULL,
    open_time TEXT,
    close_time TEXT,
    FOREIGN KEY (event_id) REFERENCES events(id)
);

-- Virtual Tripwire config — see core/tripwire.py. At most one row per
-- (booth_id, camera_id): configuring a new line for a camera that already
-- has one replaces it (UPSERT in upsert_tripwire below), matching the
-- single-line-per-camera UI ("ตั้งค่าเส้นนับคน"). x1/y1/x2/y2 are normalized
-- (0.0-1.0) against that camera's own frame, so the line survives a capture
-- resolution change.
CREATE TABLE IF NOT EXISTS tripwires (
    id TEXT PRIMARY KEY,
    booth_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    inside_side TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_ts REAL NOT NULL,
    UNIQUE (booth_id, camera_id)
);

-- One row per crossing event a TripwireCounter fires (core/tripwire.py) —
-- the durable log behind the Dashboard's visitor IN/OUT analytics. Scoped
-- by booth_id/event_id/ts exactly like every other logged-event table here,
-- so it plugs into the existing _scope_clause filtering for free.
CREATE TABLE IF NOT EXISTS tripwire_crossings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    booth_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    tripwire_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    direction TEXT NOT NULL,
    event_key TEXT
);

-- One row per Global Person identity core/reid.py's GlobalIdentityRegistry
-- has matched/created for this booth (spec sections 27/28) — a durable
-- summary for the Dashboard's "Unique People" KPI, not the matcher itself
-- (matching stays in-memory; see core/reid.py's module docstring on why).
-- global_id is only unique *within* one booth's own GlobalIdentityRegistry
-- instance, so the primary key is the (booth_id, global_id) pair, not
-- global_id alone — two different booths' registries can otherwise mint the
-- same "P000001" independently.
CREATE TABLE IF NOT EXISTS global_persons (
    booth_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    global_id TEXT NOT NULL,
    ts REAL NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    camera_count INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (booth_id, global_id)
);

-- Durable summary of core/attributes.py's GlobalPersonAttributeSmoother
-- output for one Global Person — model-predicted gender/age, never a
-- verified identity (spec: MongDee_Master_Prompt_FairFace_Age_Gender.md
-- section 15, Privacy/Data Minimization). No face image is ever stored
-- here, only the aggregated prediction + confidence + sample_count.
-- Same (booth_id, global_id) composite-key pattern as global_persons above,
-- for the same reason (global_id is only unique within one booth's own
-- registry).
CREATE TABLE IF NOT EXISTS person_attributes (
    booth_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    global_id TEXT NOT NULL,
    gender TEXT NOT NULL DEFAULT 'UNKNOWN',
    gender_confidence REAL NOT NULL DEFAULT 0,
    age_group TEXT NOT NULL DEFAULT 'UNKNOWN',
    age_category TEXT NOT NULL DEFAULT 'UNKNOWN',
    age_confidence REAL NOT NULL DEFAULT 0,
    sample_count INTEGER NOT NULL DEFAULT 0,
    ts REAL NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (booth_id, global_id)
);

-- One row per core.booth_sessions.BoothSession — a continuous ENTRY -> EXIT
-- visit (spec: MongDee_Master_Prompt_Accurate_Person_Counting_ReID.md
-- section 29 "Tripwire กับ Global Identity" + section 30 "Double Count
-- Prevention"). `person_key` is whatever BoothSessionManager was keyed by
-- (a Global Person id when Re-ID resolved the crossing's track, else a
-- per-camera fallback key — see web/booth_manager.py._update_tripwire) and
-- is never NULL; `global_person_id` duplicates it only when Re-ID actually
-- resolved it, so a query can tell a true cross-camera visit apart from a
-- Re-ID-less fallback one. Written once (INSERT) when the ENTRY opens the
-- session, then updated in place (UPDATE) when the matching EXIT closes it
-- or when Re-ID observes the same person at another camera mid-visit —
-- session_id is stable across both, unlike tripwire_crossings' per-event
-- rows. cameras_seen is a JSON array (SQLite has no native array type).
CREATE TABLE IF NOT EXISTS booth_sessions (
    session_id TEXT NOT NULL,
    booth_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    person_key TEXT NOT NULL,
    global_person_id TEXT,
    entered_at REAL NOT NULL,
    exited_at REAL,
    dwell_seconds REAL,
    cameras_seen TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    ts REAL NOT NULL,
    PRIMARY KEY (booth_id, session_id)
);
"""


def _migrate_booths_schedule(conn) -> None:
    """open_time/close_time (HH:MM, both NULL = no schedule / always on —
    today's behavior) were added after `booths` first shipped, so a DB
    created before this exists without them; `CREATE TABLE IF NOT EXISTS`
    above is a no-op against an existing table, so add the columns by hand
    here for anyone upgrading in place."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(booths)")}
    if "open_time" not in cols:
        conn.execute("ALTER TABLE booths ADD COLUMN open_time TEXT")
    if "close_time" not in cols:
        conn.execute("ALTER TABLE booths ADD COLUMN close_time TEXT")


def _migrate_presence_category(conn) -> None:
    """category ('child'/'female'/'male'/'unknown', NULL for a session
    logged before this existed or with no gender_age_backend configured)
    was added after `presence_sessions` first shipped — same
    ALTER-TABLE-by-hand situation as _migrate_booths_schedule above."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(presence_sessions)")}
    if "category" not in cols:
        conn.execute("ALTER TABLE presence_sessions ADD COLUMN category TEXT")


def _migrate_tripwire_crossing_idempotency(conn) -> None:
    """event_key (spec section 17: crossing-event idempotency) was added
    after `tripwire_crossings` first shipped. A plain (non-UNIQUE-in-schema)
    column plus a separate `CREATE UNIQUE INDEX IF NOT EXISTS` — rather than
    declaring UNIQUE inline on the column — so this same migration works
    identically whether the table is brand new or pre-existing (ALTER TABLE
    ADD COLUMN can't retroactively add a UNIQUE constraint on an existing
    SQLite table). NULL event_keys (crossings logged before this existed)
    never collide with each other under a UNIQUE index — SQLite treats every
    NULL as distinct."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tripwire_crossings)")}
    if "event_key" not in cols:
        conn.execute("ALTER TABLE tripwire_crossings ADD COLUMN event_key TEXT")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_tripwire_crossings_event_key "
        "ON tripwire_crossings(event_key)"
    )


def _migrate_tripwire_crossing_attribute_snapshot(conn) -> None:
    """attribute_snapshot_json (spec: MongDee_Master_Prompt_FairFace_Age_
    Gender.md section 13 — "เมื่อคนผ่าน tripwire ให้ event สามารถเก็บ
    attribute snapshot ได้") was added after `tripwire_crossings` first
    shipped. Nullable and purely additive: a crossing logged with no
    Re-ID/attribute system configured (or before this existed) just has
    NULL here, and one crossing is always still exactly one row — this
    column never affects IN/OUT counting."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tripwire_crossings)")}
    if "attribute_snapshot_json" not in cols:
        conn.execute("ALTER TABLE tripwire_crossings ADD COLUMN attribute_snapshot_json TEXT")


def _migrate_tripwire_crossing_global_person(conn) -> None:
    """global_person_id (spec: MongDee_Master_Prompt_Accurate_Person_
    Counting_ReID.md section 29 "Tripwire กับ Global Identity" — a
    crossing's local_track_id must be mapped to its Global Person id, not
    left as a camera-local number only) was added after `tripwire_crossings`
    first shipped. Nullable: a crossing logged before this existed, or with
    Re-ID disabled/not-yet-resolved for that track at crossing time, just
    has NULL here — this column never affects IN/OUT counting, only which
    identity a crossing is attributed to for booth-session/dwell purposes."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tripwire_crossings)")}
    if "global_person_id" not in cols:
        conn.execute("ALTER TABLE tripwire_crossings ADD COLUMN global_person_id TEXT")


def _migrate_product_hold_event_attribution(conn) -> None:
    """global_person_id/gender/gender_confidence/age_category/age_confidence
    (spec: MongDee person-gender/age master prompt sections 22/26/27) were
    added after `product_hold_events` first shipped — same by-hand
    ALTER-TABLE situation as the other _migrate_* functions above."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(product_hold_events)")}
    for col, decl in (
        ("global_person_id", "TEXT"), ("gender", "TEXT"), ("gender_confidence", "REAL"),
        ("age_category", "TEXT"), ("age_confidence", "REAL"),
    ):
        if col not in cols:
            conn.execute(f"ALTER TABLE product_hold_events ADD COLUMN {col} {decl}")


def _migrate_product_hold_event_lifecycle(conn) -> None:
    """status/confirmed_at (spec: MongDee person-gender/age master
    continuation prompt sections 20/23 — confirm-at-2-seconds, finalize-at-
    release) were added after `product_hold_events` first shipped. Every
    pre-existing row was inserted with a real, already-known hold_end_ts/
    duration_sec (the old insert-at-release behavior), so DEFAULT
    'finalized' is correct for them without touching a single row —
    ALTER TABLE ... ADD COLUMN ... DEFAULT applies retroactively in SQLite."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(product_hold_events)")}
    if "status" not in cols:
        conn.execute("ALTER TABLE product_hold_events ADD COLUMN status TEXT NOT NULL DEFAULT 'finalized'")
    if "confirmed_at" not in cols:
        conn.execute("ALTER TABLE product_hold_events ADD COLUMN confirmed_at REAL")


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    _assert_not_production_under_test(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        # WAL mode is a one-time, persisted-in-the-file setting — worth
        # doing here rather than per-connection. Without it, this app's own
        # documented usage pattern (README: web server + desktop booth +
        # Dashboard all open at once, possibly across processes) means any
        # write from one process holds an exclusive lock on the *entire*
        # file under the default rollback-journal mode, stalling every
        # other process's reads until it commits. With several camera
        # threads logging heartbeats/health events/presence sessions
        # continuously, that easily adds up to Dashboard (or a second booth
        # process) hanging for many seconds on startup for no visible
        # reason. WAL lets readers proceed concurrently with a writer.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        _migrate_booths_schedule(conn)
        _migrate_presence_category(conn)
        _migrate_tripwire_crossing_idempotency(conn)
        _migrate_tripwire_crossing_attribute_snapshot(conn)
        _migrate_tripwire_crossing_global_person(conn)
        _migrate_product_hold_event_attribution(conn)
        _migrate_product_hold_event_lifecycle(conn)


def _connect(db_path: Path):
    _assert_not_production_under_test(db_path)
    return sqlite3.connect(db_path, timeout=5)


def log_interaction(db_path, booth_id, event_id, camera_id, product_key,
                     product_name, confidence, question=None, answer=None):
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO interactions "
            "(ts, booth_id, event_id, camera_id, product_key, product_name, "
            "confidence, question, answer) VALUES (?,?,?,?,?,?,?,?,?)",
            (time.time(), booth_id, event_id, camera_id, product_key,
             product_name, confidence, question, answer),
        )


def log_health_event(db_path, booth_id, event_id, camera_id, component, status, message=None):
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO health_events "
            "(ts, booth_id, event_id, camera_id, component, status, message) "
            "VALUES (?,?,?,?,?,?,?)",
            (time.time(), booth_id, event_id, camera_id, component, status, message),
        )


def log_readiness_check(db_path, booth_id, event_id, overall_status, detail_json):
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO readiness_checks "
            "(ts, booth_id, event_id, overall_status, detail_json) VALUES (?,?,?,?,?)",
            (time.time(), booth_id, event_id, overall_status, detail_json),
        )


def log_heartbeat(db_path, booth_id, event_id, status, active_cameras):
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO heartbeats (ts, booth_id, event_id, status, active_cameras) "
            "VALUES (?,?,?,?,?)",
            (time.time(), booth_id, event_id, status, active_cameras),
        )


def log_product_hold_event(db_path, booth_id, event_id, camera_id, product_key, product_name,
                            holder_track_id, hold_start_ts, hold_end_ts, duration_sec,
                            global_person_id=None, gender=None, gender_confidence=None,
                            age_category=None, age_confidence=None):
    """Spec: MongDee person-gender/age master prompt section 22 (Interest
    Event Schema) + sections 26/27 (Gender/Age Attribution) — see the
    `product_hold_events` table's own comment for what the five attribution
    columns mean and why they're a snapshot, never a live/rewritable value."""
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO product_hold_events "
            "(ts, booth_id, event_id, camera_id, product_key, product_name, "
            "holder_track_id, hold_start_ts, hold_end_ts, duration_sec, "
            "global_person_id, gender, gender_confidence, age_category, age_confidence) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), booth_id, event_id, camera_id, product_key, product_name,
             holder_track_id, hold_start_ts, hold_end_ts, duration_sec,
             global_person_id, gender, gender_confidence, age_category, age_confidence),
        )


def open_product_interest_event(db_path, booth_id, event_id, camera_id, product_key, product_name,
                                 holder_track_id, hold_start_ts, confirmed_at,
                                 global_person_id=None, gender=None, gender_confidence=None,
                                 age_category=None, age_confidence=None) -> int:
    """Writes the row the instant a hold is confirmed (spec: MongDee
    person-gender/age master continuation prompt section 20 — "ยืนยัน
    Interest ตอนครบ 2 seconds ไม่ต้องรอ release"), not at release.
    hold_end_ts/duration_sec are provisional (set to confirmed_at / the
    elapsed time at confirmation) only because both columns predate this
    feature as NOT NULL, and a safe additive migration can't relax that
    without a full table rebuild (spec section 37: no destructive
    migrations) — `status='confirmed'` is the real signal they're not final
    yet. finalize_product_interest_event overwrites them with the true
    values once the hold actually ends.

    gender/age/global_person_id are written once, here, and never touched
    again (spec section 26: a snapshot at confirmation, never rewritten
    later even if the Global Person's profile is subsequently refined).
    Returns the new row's id, for finalize_product_interest_event to target."""
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO product_hold_events "
            "(ts, booth_id, event_id, camera_id, product_key, product_name, holder_track_id, "
            "hold_start_ts, hold_end_ts, duration_sec, global_person_id, gender, gender_confidence, "
            "age_category, age_confidence, status, confirmed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), booth_id, event_id, camera_id, product_key, product_name, holder_track_id,
             hold_start_ts, confirmed_at, confirmed_at - hold_start_ts,
             global_person_id, gender, gender_confidence, age_category, age_confidence,
             "confirmed", confirmed_at),
        )
        return cur.lastrowid


def finalize_product_interest_event(db_path, row_id, hold_end_ts, duration_sec) -> None:
    """Overwrites only the end-of-interaction fields once a confirmed hold
    actually ends (spec section 23: release finalizes the same interaction,
    it never creates a second event) — gender/age/global_person_id are
    untouched, exactly per open_product_interest_event's docstring."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE product_hold_events SET hold_end_ts=?, duration_sec=?, status='finalized' WHERE id=?",
            (hold_end_ts, duration_sec, row_id),
        )


def log_presence_session(db_path, booth_id, event_id, camera_id, track_id,
                          start_ts, end_ts, duration_sec, category=None):
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO presence_sessions "
            "(ts, booth_id, event_id, camera_id, track_id, start_ts, end_ts, duration_sec, category) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (time.time(), booth_id, event_id, camera_id, track_id, start_ts, end_ts, duration_sec, category),
        )


# --------------------------------------------------------------- tripwires

def upsert_tripwire(db_path, tripwire_id, booth_id, camera_id, x1, y1, x2, y2,
                     inside_side, enabled=True):
    """Create or replace the one tripwire configured for (booth_id,
    camera_id) — see the `tripwires` table's UNIQUE constraint. Setting a
    new line for a camera that already has one is a deliberate replace, not
    an error, matching the single-line-per-camera "ตั้งค่าเส้นนับคน" UI."""
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO tripwires (id, booth_id, camera_id, x1, y1, x2, y2, inside_side, enabled, created_ts) "
            "VALUES (?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(booth_id, camera_id) DO UPDATE SET "
            "id=excluded.id, x1=excluded.x1, y1=excluded.y1, x2=excluded.x2, y2=excluded.y2, "
            "inside_side=excluded.inside_side, enabled=excluded.enabled",
            (tripwire_id, booth_id, camera_id, x1, y1, x2, y2, inside_side, int(enabled), time.time()),
        )


def get_tripwire(db_path, booth_id, camera_id) -> dict | None:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM tripwires WHERE booth_id = ? AND camera_id = ?", (booth_id, camera_id)
        ).fetchone()
    return dict(row) if row else None


def list_tripwires(db_path, booth_id) -> list[dict]:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM tripwires WHERE booth_id = ?", (booth_id,)).fetchall()
    return [dict(r) for r in rows]


def delete_tripwire(db_path, booth_id, camera_id) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM tripwires WHERE booth_id = ? AND camera_id = ?", (booth_id, camera_id))


def log_tripwire_crossing(db_path, booth_id, event_id, camera_id, tripwire_id, track_id, direction, ts=None,
                           attribute_snapshot: dict | None = None, global_person_id: str | None = None):
    """Spec section 17 (idempotency): a retry, websocket reconnect, or
    duplicate processing of the exact same (camera, track, tripwire,
    direction, ts) crossing must never insert a second row. ts already
    comes from TripwireCounter.update()'s own `now` at the moment the
    crossing fired, so re-delivering that same in-memory event dict always
    reproduces the same event_key — INSERT OR IGNORE against the unique
    index silently no-ops on the duplicate instead of double-counting.

    attribute_snapshot (spec: MongDee_Master_Prompt_FairFace_Age_Gender.md
    section 13) is an optional read-only copy of whatever
    core.attributes.GlobalPersonAttributeSmoother already knew about this
    crossing's Global Person at the moment it fired — attaching it never
    creates a second crossing row or re-triggers counting; the (camera,
    track, tripwire, direction, ts) identity above is unchanged either way.

    global_person_id (spec: MongDee_Master_Prompt_Accurate_Person_Counting_
    ReID.md section 29) is core.reid.GlobalIdentityRegistry's resolved
    identity for this crossing's (camera_id, track_id) at the moment it
    fired, or None when Re-ID is disabled or hadn't resolved that track yet
    — an honest "unknown identity", never a guess."""
    ts = ts if ts is not None else time.time()
    event_key = f"{camera_id}:{tripwire_id}:{track_id}:{direction}:{ts!r}"
    snapshot_json = json.dumps(attribute_snapshot, ensure_ascii=False) if attribute_snapshot else None
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO tripwire_crossings "
            "(ts, booth_id, event_id, camera_id, tripwire_id, track_id, direction, event_key, "
            "attribute_snapshot_json, global_person_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ts, booth_id, event_id, camera_id, tripwire_id, track_id, direction, event_key, snapshot_json,
             global_person_id),
        )


def _rows_as_dicts(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _day_range_ts(date_str: str) -> tuple[float, float]:
    """[start, end) unix ts spanning one local-calendar day, e.g.
    "2026-09-15" -> (that midnight, next midnight) — the Dashboard's day
    picker filters every query to one of these windows."""
    start = time.mktime(datetime.strptime(date_str, "%Y-%m-%d").timetuple())
    return start, start + 86400


def _scope_clause(event_id=None, booth_id=None, date=None, camera_id=None) -> tuple[str, list]:
    """Common `WHERE ... AND <this>` fragment (+ its params) every
    dashboard query filters by: event_id, booth_id, and optionally one
    local-calendar day. Every table these apply to has an `event_id`,
    `booth_id`, and `ts` column, so the same fragment drops into any of
    their `WHERE 1=1` queries unchanged.

    camera_id (spec: MongDee person-gender/age master continuation prompt
    section 42) is opt-in and defaults to None/no-filter — only tables that
    actually have a `camera_id` column (product_hold_events, tripwire_
    crossings, presence_sessions, ...) should ever pass it; every existing
    caller that doesn't pass it is completely unaffected."""
    clauses = []
    params: list = []
    if event_id:
        clauses.append("event_id = ?")
        params.append(event_id)
    if booth_id:
        clauses.append("booth_id = ?")
        params.append(booth_id)
    if camera_id:
        clauses.append("camera_id = ?")
        params.append(camera_id)
    if date:
        start_ts, end_ts = _day_range_ts(date)
        clauses.append("ts >= ? AND ts < ?")
        params.extend([start_ts, end_ts])
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def query_interactions(db_path, event_id=None, booth_id=None, date=None, limit=200):
    clause, params = _scope_clause(event_id, booth_id, date)
    sql = "SELECT * FROM interactions WHERE 1=1" + clause + " ORDER BY ts DESC LIMIT ?"
    with _connect(db_path) as conn:
        return _rows_as_dicts(conn.execute(sql, params + [limit]))


def query_top_products(db_path, event_id=None, booth_id=None, date=None, limit=10):
    clause, params = _scope_clause(event_id, booth_id, date)
    sql = (
        "SELECT product_name, COUNT(*) as views FROM interactions WHERE 1=1" + clause +
        " GROUP BY product_name ORDER BY views DESC LIMIT ?"
    )
    with _connect(db_path) as conn:
        return _rows_as_dicts(conn.execute(sql, params + [limit]))


def query_product_movers(db_path, event_id=None, booth_id=None, date=None, limit=10):
    """Which products have been picked up/moved by the most (ephemeral)
    people, ranked — each product_hold_events row is one person's one
    continuous hold, so COUNT(*) is the mover count (spec: MongDee
    person-gender/age master prompt section 33 "Product Popularity").

    unique_persons (section 34) is a *separate* number from mover_count —
    COUNT(DISTINCT global_person_id) is only the visitors Re-ID actually
    resolved for that hold; a hold logged with no resolved identity
    (Re-ID disabled, or not yet resolved at hold time) contributes to
    mover_count but is excluded here rather than guessed at, so
    unique_persons is always <= mover_count, never a fabricated equal-or-
    higher number."""
    clause, params = _scope_clause(event_id, booth_id, date)
    sql = (
        "SELECT product_name, COUNT(*) as mover_count, SUM(duration_sec) as total_interest_sec, "
        "COUNT(DISTINCT global_person_id) as unique_persons "
        "FROM product_hold_events WHERE 1=1" + clause +
        " GROUP BY product_name ORDER BY mover_count DESC LIMIT ?"
    )
    with _connect(db_path) as conn:
        return _rows_as_dicts(conn.execute(sql, params + [limit]))


def query_product_hold_events(db_path, event_id=None, booth_id=None, date=None, limit=2000):
    """Raw history of every pick-up/hold, newest first — the underlying
    events behind query_product_movers()'s aggregate counts, kept queryable
    on their own for later analysis (not just the summarized totals). Also
    what the Dashboard's hourly interest chart buckets client-side (spec
    section 41) — limit matches query_tripwire_crossings's own default for
    the same reason: a busy day's full 24h picture must not get truncated
    by a small default page size."""
    clause, params = _scope_clause(event_id, booth_id, date)
    sql = "SELECT * FROM product_hold_events WHERE 1=1" + clause + " ORDER BY ts DESC LIMIT ?"
    with _connect(db_path) as conn:
        return _rows_as_dicts(conn.execute(sql, params + [limit]))


# ------------------------------------------------------- product interest analytics
# Spec: MongDee person-gender/age master prompt sections 28-45 — Gender x
# Product / Age x Product / popularity / zero-interest / gender+age totals.
# Every function here reads only product_hold_events (never re-derives
# anything from live state) and is additive to query_product_movers/
# query_product_hold_events above, not a replacement for either.

def query_product_interest_summary(db_path, event_id=None, booth_id=None, date=None,
                                    camera_id=None) -> list[dict]:
    """Per product: interest_count (every logged hold) + unique_persons
    (distinct resolved Global Persons — see query_product_movers's own
    docstring on why this is never >= interest_count by guessing). Spec
    sections 33/34/42. Only products with >=1 hold event appear — the
    catalog itself (products.json via core.products) lives outside this
    database, so a caller wanting section 37's "products with zero
    interest" combines this with the full catalog list (e.g. GET
    /api/products) rather than this function inventing catalog rows."""
    clause, params = _scope_clause(event_id, booth_id, date, camera_id)
    sql = (
        "SELECT product_key, product_name, COUNT(*) as interest_count, "
        "COUNT(DISTINCT global_person_id) as unique_persons, "
        "SUM(duration_sec) as total_interest_sec "
        "FROM product_hold_events WHERE 1=1" + clause +
        " GROUP BY product_key, product_name ORDER BY interest_count DESC"
    )
    with _connect(db_path) as conn:
        return _rows_as_dicts(conn.execute(sql, params))


def query_product_gender_matrix(db_path, event_id=None, booth_id=None, date=None,
                                 camera_id=None) -> list[dict]:
    """[{"product_key", "product_name", "MALE", "FEMALE", "UNKNOWN"}, ...] —
    spec sections 28/35/43 (Gender x Product). A hold with no resolved
    gender (Re-ID/FairFace not configured, or not yet resolved at hold
    time) counts as UNKNOWN, same convention as query_attribute_breakdown."""
    clause, params = _scope_clause(event_id, booth_id, date, camera_id)
    sql = (
        "SELECT product_key, product_name, COALESCE(gender, 'UNKNOWN') as gender, COUNT(*) as n "
        "FROM product_hold_events WHERE 1=1" + clause +
        " GROUP BY product_key, product_name, gender"
    )
    with _connect(db_path) as conn:
        rows = _rows_as_dicts(conn.execute(sql, params))
    by_product: dict[str, dict] = {}
    for row in rows:
        key = row["product_key"]
        entry = by_product.setdefault(key, {
            "product_key": key, "product_name": row["product_name"],
            "MALE": 0, "FEMALE": 0, "UNKNOWN": 0,
        })
        entry[row["gender"]] = entry.get(row["gender"], 0) + row["n"]
    return sorted(by_product.values(), key=lambda e: e["product_name"])


def query_product_age_matrix(db_path, event_id=None, booth_id=None, date=None,
                              camera_id=None) -> list[dict]:
    """Same shape as query_product_gender_matrix, keyed by age_category
    (CHILD/ADULT/UNKNOWN) instead of gender — spec sections 29/36/44."""
    clause, params = _scope_clause(event_id, booth_id, date, camera_id)
    sql = (
        "SELECT product_key, product_name, COALESCE(age_category, 'UNKNOWN') as age_category, COUNT(*) as n "
        "FROM product_hold_events WHERE 1=1" + clause +
        " GROUP BY product_key, product_name, age_category"
    )
    with _connect(db_path) as conn:
        rows = _rows_as_dicts(conn.execute(sql, params))
    by_product: dict[str, dict] = {}
    for row in rows:
        key = row["product_key"]
        entry = by_product.setdefault(key, {
            "product_key": key, "product_name": row["product_name"],
            "CHILD": 0, "ADULT": 0, "UNKNOWN": 0,
        })
        entry[row["age_category"]] = entry.get(row["age_category"], 0) + row["n"]
    return sorted(by_product.values(), key=lambda e: e["product_name"])


def query_gender_interest_totals(db_path, event_id=None, booth_id=None, date=None,
                                  camera_id=None) -> dict:
    """Total confirmed interest (hold) events by gender, across every
    product — spec sections 31/40 ("เพศไหนสนใจสินค้ามากที่สุด")."""
    clause, params = _scope_clause(event_id, booth_id, date, camera_id)
    sql = ("SELECT COALESCE(gender, 'UNKNOWN') as gender, COUNT(*) as n "
           "FROM product_hold_events WHERE 1=1" + clause + " GROUP BY gender")
    with _connect(db_path) as conn:
        rows = _rows_as_dicts(conn.execute(sql, params))
    totals = {"MALE": 0, "FEMALE": 0, "UNKNOWN": 0}
    for row in rows:
        totals[row["gender"]] = totals.get(row["gender"], 0) + row["n"]
    return totals


def query_age_interest_totals(db_path, event_id=None, booth_id=None, date=None,
                               camera_id=None) -> dict:
    """Total confirmed interest (hold) events by age_category, across every
    product — spec sections 32/41 ("ช่วงอายุไหนสนใจสินค้ามากที่สุด")."""
    clause, params = _scope_clause(event_id, booth_id, date, camera_id)
    sql = ("SELECT COALESCE(age_category, 'UNKNOWN') as age_category, COUNT(*) as n "
           "FROM product_hold_events WHERE 1=1" + clause + " GROUP BY age_category")
    with _connect(db_path) as conn:
        rows = _rows_as_dicts(conn.execute(sql, params))
    totals = {"CHILD": 0, "ADULT": 0, "UNKNOWN": 0}
    for row in rows:
        totals[row["age_category"]] = totals.get(row["age_category"], 0) + row["n"]
    return totals


def query_interest_overview(db_path, event_id=None, booth_id=None, date=None, camera_id=None) -> dict:
    """Headline KPIs for the Dashboard's Product Interest Overview (spec
    section 39): total interest events logged, and how many *distinct*
    Global Persons they came from (never conflated — see spec section 34)."""
    clause, params = _scope_clause(event_id, booth_id, date, camera_id)
    sql = ("SELECT COUNT(*) as total, COUNT(DISTINCT global_person_id) as unique_persons "
           "FROM product_hold_events WHERE 1=1" + clause)
    with _connect(db_path) as conn:
        row = conn.execute(sql, params).fetchone()
    return {"total_interest_events": row[0], "unique_interested_persons": row[1]}


def query_presence_sessions(db_path, event_id=None, booth_id=None, date=None, limit=200):
    clause, params = _scope_clause(event_id, booth_id, date)
    sql = "SELECT * FROM presence_sessions WHERE 1=1" + clause + " ORDER BY ts DESC LIMIT ?"
    with _connect(db_path) as conn:
        return _rows_as_dicts(conn.execute(sql, params + [limit]))


def query_presence_stats(db_path, event_id=None, booth_id=None, date=None):
    sessions = query_presence_sessions(db_path, event_id, booth_id, date, limit=100000)
    durations = [row["duration_sec"] for row in sessions]
    if not durations:
        return {"count": 0, "avg_sec": 0.0, "min_sec": 0.0, "max_sec": 0.0}
    return {
        "count": len(durations),
        "avg_sec": sum(durations) / len(durations),
        "min_sec": min(durations),
        "max_sec": max(durations),
    }


def query_presence_breakdown(db_path, event_id=None, booth_id=None, date=None) -> dict:
    """Visit counts by category ('child'/'female'/'male'/'unknown') for the
    Dashboard's gender/age breakdown — 'unknown' covers both sessions
    logged before `category` existed (NULL) and sessions where no category
    was ever confidently determined (no --gender-model configured, or the
    person was never classified above the confidence floor)."""
    clause, params = _scope_clause(event_id, booth_id, date)
    sql = "SELECT category, COUNT(*) as n FROM presence_sessions WHERE 1=1" + clause + " GROUP BY category"
    with _connect(db_path) as conn:
        rows = _rows_as_dicts(conn.execute(sql, params))
    counts = {"child": 0, "female": 0, "male": 0, "unknown": 0}
    for row in rows:
        cat = row["category"] or "unknown"
        counts[cat] = counts.get(cat, 0) + row["n"]
    return counts


def query_tripwire_crossings(db_path, event_id=None, booth_id=None, date=None, limit=2000):
    """Raw crossing history, oldest first (the Dashboard buckets this
    client-side into an IN/OUT timeline the same way it already does for
    interactions/presence sessions — see web/static/dashboard.js)."""
    clause, params = _scope_clause(event_id, booth_id, date)
    sql = "SELECT * FROM tripwire_crossings WHERE 1=1" + clause + " ORDER BY ts ASC LIMIT ?"
    with _connect(db_path) as conn:
        return _rows_as_dicts(conn.execute(sql, params + [limit]))


def query_tripwire_stats(db_path, event_id=None, booth_id=None, date=None) -> dict:
    """Aggregate visitor counters for the Dashboard's KPI cards: total IN,
    total OUT, current occupancy (IN minus OUT, floored at 0), and the peak
    simultaneous occupancy reached anywhere in the filtered scope. Peak is
    computed by replaying every crossing in chronological order in Python
    (same pattern as query_presence_stats above) rather than in SQL — a
    plain SUM/COUNT can't express "running balance, floored, at its max"."""
    crossings = query_tripwire_crossings(db_path, event_id, booth_id, date, limit=1000000)
    total_in = sum(1 for c in crossings if c["direction"] == "in")
    total_out = sum(1 for c in crossings if c["direction"] == "out")
    balance = 0
    peak = 0
    for c in crossings:
        balance += 1 if c["direction"] == "in" else -1
        if balance < 0:
            balance = 0
        peak = max(peak, balance)
    return {
        "total_in": total_in,
        "total_out": total_out,
        "current_inside": max(0, total_in - total_out),
        "peak_inside": peak,
    }


# ------------------------------------------------------------ booth sessions

def open_booth_session(db_path, booth_id, event_id, session_id, person_key, global_person_id,
                        entered_at, cameras_seen):
    """One row per core.booth_sessions.BoothSession, written the moment
    web/booth_manager.py's tripwire handling turns an ENTRY crossing into a
    newly-opened session (spec section 29). `global_person_id` is the
    Multi-Camera Re-ID identity when Re-ID is enabled and already resolved
    for the crossing track at that moment, else NULL — `person_key` (never
    NULL) is what BoothSessionManager actually keys sessions by, and falls
    back to a per-camera-local key when Re-ID can't identify the track yet,
    so dwell tracking still works with Re-ID disabled/not-yet-resolved."""
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO booth_sessions "
            "(session_id, booth_id, event_id, person_key, global_person_id, entered_at, "
            "exited_at, dwell_seconds, cameras_seen, status, ts) "
            "VALUES (?,?,?,?,?,?,NULL,NULL,?,'active',?)",
            (session_id, booth_id, event_id, person_key, global_person_id, entered_at,
             json.dumps(sorted(cameras_seen), ensure_ascii=False), entered_at),
        )


def close_booth_session(db_path, booth_id, session_id, exited_at, dwell_seconds, cameras_seen):
    """Marks a booth session closed once its matching EXIT crossing fires.
    dwell_seconds always comes from real entered_at/exited_at timestamps
    (core.booth_sessions.BoothSessionManager.on_exit), never frame counts."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE booth_sessions SET exited_at=?, dwell_seconds=?, cameras_seen=?, status='closed' "
            "WHERE booth_id=? AND session_id=?",
            (exited_at, dwell_seconds, json.dumps(sorted(cameras_seen), ensure_ascii=False),
             booth_id, session_id),
        )


def update_booth_session_cameras(db_path, booth_id, session_id, cameras_seen):
    """Called when core.booth_sessions.BoothSessionManager.note_camera_seen
    reports a *new* camera was added to an already-open session (e.g. Re-ID
    matched the same Global Person at a second camera mid-visit, with no
    tripwire crossing there at all — spec section 21's "person switches
    camera without a second tripwire" case). WHERE status='active'
    defensively skips a no-op write if the session already closed in the
    same race."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE booth_sessions SET cameras_seen=? WHERE booth_id=? AND session_id=? AND status='active'",
            (json.dumps(sorted(cameras_seen), ensure_ascii=False), booth_id, session_id),
        )


def query_booth_sessions(db_path, event_id=None, booth_id=None, date=None, limit=200):
    """Newest-first history of booth visits (open and closed), for the
    Dashboard's per-person "entry time / dwell / exit time" list (spec
    section 36)."""
    clause, params = _scope_clause(event_id, booth_id, date)
    sql = "SELECT * FROM booth_sessions WHERE 1=1" + clause + " ORDER BY entered_at DESC LIMIT ?"
    with _connect(db_path) as conn:
        rows = _rows_as_dicts(conn.execute(sql, params + [limit]))
    for row in rows:
        row["cameras_seen"] = json.loads(row["cameras_seen"]) if row["cameras_seen"] else []
    return rows


def query_booth_session_stats(db_path, event_id=None, booth_id=None, date=None) -> dict:
    """Aggregate dwell-time KPIs for the Dashboard: how many visits are
    currently open, and avg/min/max dwell of the closed ones in scope (spec
    section 36: "Average Dwell Time" / "Current Active Sessions" /
    "Longest Dwell")."""
    clause, params = _scope_clause(event_id, booth_id, date)
    sql = "SELECT status, dwell_seconds FROM booth_sessions WHERE 1=1" + clause
    with _connect(db_path) as conn:
        rows = _rows_as_dicts(conn.execute(sql, params))
    durations = [r["dwell_seconds"] for r in rows if r["dwell_seconds"] is not None]
    return {
        "total_sessions": len(rows),
        "active_count": sum(1 for r in rows if r["status"] == "active"),
        "avg_dwell_sec": sum(durations) / len(durations) if durations else 0.0,
        "min_dwell_sec": min(durations) if durations else 0.0,
        "max_dwell_sec": max(durations) if durations else 0.0,
    }


# ----------------------------------------------------------- global persons

def upsert_global_person(db_path, booth_id, event_id, global_id, first_seen, last_seen, camera_count):
    """Durable summary row for one core.reid.GlobalPerson, written by
    web/booth_manager.py whenever its in-memory GlobalIdentityRegistry
    creates or refines an identity (spec section 26: DB is a persistence/
    analytics layer, never the per-frame matcher itself)."""
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO global_persons (booth_id, event_id, global_id, ts, first_seen, last_seen, camera_count) "
            "VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(booth_id, global_id) DO UPDATE SET "
            "last_seen=excluded.last_seen, camera_count=excluded.camera_count",
            (booth_id, event_id, global_id, first_seen, first_seen, last_seen, camera_count),
        )


MERGED_ID_PREFIX = "merged:"     # identities folded into another one keep their rows, renamed, and are not counted


def merge_global_person(db_path, booth_id, loser_id, winner_id) -> None:
    """Re-ID decided that `loser_id` and `winner_id` are one physical person. Everything that pointed at the
    loser now points at the winner, and the loser's summary rows are kept but renamed (MERGED_ID_PREFIX) so they
    are no longer counted as a separate visitor."""
    if loser_id == winner_id:
        return
    renamed = MERGED_ID_PREFIX + loser_id
    with _connect(db_path) as conn:
        for table in ("product_hold_events", "tripwire_crossings", "booth_sessions"):
            conn.execute(f"UPDATE {table} SET global_person_id=? WHERE booth_id=? AND global_person_id=?",
                         (winner_id, booth_id, loser_id))
        conn.execute("UPDATE booth_sessions SET person_key=? WHERE booth_id=? AND person_key=?",
                     (winner_id, booth_id, loser_id))
        row = conn.execute("SELECT first_seen, last_seen, camera_count FROM global_persons "
                           "WHERE booth_id=? AND global_id=?", (booth_id, loser_id)).fetchone()
        if row is not None:
            conn.execute("UPDATE global_persons SET first_seen=MIN(first_seen, ?), last_seen=MAX(last_seen, ?), "
                         "camera_count=MAX(camera_count, ?) WHERE booth_id=? AND global_id=?",
                         (row[0], row[1], row[2], booth_id, winner_id))
            conn.execute("DELETE FROM global_persons WHERE booth_id=? AND global_id=? AND EXISTS "
                         "(SELECT 1 FROM global_persons WHERE booth_id=? AND global_id=?)",
                         (booth_id, renamed, booth_id, renamed))
            conn.execute("UPDATE global_persons SET global_id=? WHERE booth_id=? AND global_id=?",
                         (renamed, booth_id, loser_id))
        conn.execute("DELETE FROM person_attributes WHERE booth_id=? AND global_id=? AND EXISTS "
                     "(SELECT 1 FROM person_attributes WHERE booth_id=? AND global_id=?)",
                     (booth_id, renamed, booth_id, renamed))
        conn.execute("UPDATE person_attributes SET global_id=? WHERE booth_id=? AND global_id=?",
                     (renamed, booth_id, loser_id))


def query_unique_people_count(db_path, event_id=None, booth_id=None, date=None) -> int:
    """Spec section 28: distinct from `Current Visible People`/Tripwire IN
    counts — this is COUNT(DISTINCT global identity), never a raw per-camera
    detection tally."""
    clause, params = _scope_clause(event_id, booth_id, date)
    sql = "SELECT COUNT(*) FROM global_persons WHERE global_id NOT LIKE '" + MERGED_ID_PREFIX + "%'" + clause
    with _connect(db_path) as conn:
        return conn.execute(sql, params).fetchone()[0]


# --------------------------------------------------------- person attributes

def upsert_person_attributes(db_path, booth_id, event_id, global_id, gender, gender_confidence,
                              age_group, age_category, age_confidence, sample_count,
                              first_seen, last_seen):
    """Durable summary row for one core.attributes.AttributeResult, written
    by web/booth_manager.py whenever its in-memory
    GlobalPersonAttributeSmoother refines a Global Person's profile (spec
    section 9: DB is a persistence layer, the smoother's own in-memory
    aggregate is the hot path — this is never queried per-frame). Never
    stores a face image (spec section 15)."""
    now = time.time()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO person_attributes "
            "(booth_id, event_id, global_id, gender, gender_confidence, age_group, age_category, "
            "age_confidence, sample_count, ts, first_seen, last_seen, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(booth_id, global_id) DO UPDATE SET "
            "gender=excluded.gender, gender_confidence=excluded.gender_confidence, "
            "age_group=excluded.age_group, age_category=excluded.age_category, "
            "age_confidence=excluded.age_confidence, sample_count=excluded.sample_count, "
            "last_seen=excluded.last_seen, updated_at=excluded.updated_at",
            (booth_id, event_id, global_id, gender, gender_confidence, age_group, age_category,
             age_confidence, sample_count, first_seen, first_seen, last_seen, now, now),
        )


def query_attribute_breakdown(db_path, event_id=None, booth_id=None, date=None) -> dict:
    """Spec section 17: gender/age-category counts of *distinct Global
    Persons*, kept separate from Tripwire IN/OUT and Unique Global Persons —
    never substituted for either. UNKNOWN is always its own bucket, never
    dropped or folded into a guess."""
    clause, params = _scope_clause(event_id, booth_id, date)
    sql = "SELECT gender, age_category, COUNT(*) as n FROM person_attributes WHERE global_id NOT LIKE '" + \
          MERGED_ID_PREFIX + "%'" + clause + " GROUP BY gender, age_category"
    with _connect(db_path) as conn:
        rows = _rows_as_dicts(conn.execute(sql, params))
    gender_counts = {"MALE": 0, "FEMALE": 0, "UNKNOWN": 0}
    age_counts = {"CHILD": 0, "ADULT": 0, "UNKNOWN": 0}
    for row in rows:
        gender_counts[row["gender"]] = gender_counts.get(row["gender"], 0) + row["n"]
        age_counts[row["age_category"]] = age_counts.get(row["age_category"], 0) + row["n"]
    return {"gender": gender_counts, "age_category": age_counts}


def query_health_events(db_path, event_id=None, booth_id=None, date=None, limit=100):
    clause, params = _scope_clause(event_id, booth_id, date)
    sql = "SELECT * FROM health_events WHERE 1=1" + clause + " ORDER BY ts DESC LIMIT ?"
    with _connect(db_path) as conn:
        return _rows_as_dicts(conn.execute(sql, params + [limit]))


def query_available_dates(db_path, event_id=None, booth_id=None) -> list[str]:
    """Distinct local-calendar days that have any logged activity
    (interactions/health/presence/hold events), newest first — populates
    the Dashboard's day picker so it only ever lists days with real data."""
    clause, params = _scope_clause(event_id, booth_id)
    dates: set[str] = set()
    with _connect(db_path) as conn:
        for table in ("interactions", "health_events", "presence_sessions", "product_hold_events",
                      "tripwire_crossings", "global_persons", "person_attributes", "booth_sessions"):
            sql = f"SELECT DISTINCT date(ts, 'unixepoch', 'localtime') AS d FROM {table} WHERE 1=1{clause}"
            dates.update(row[0] for row in conn.execute(sql, params) if row[0])
    return sorted(dates, reverse=True)


SCOPED_TABLES = ("interactions", "health_events", "readiness_checks", "heartbeats",
                 "product_hold_events", "presence_sessions", "tripwire_crossings", "global_persons",
                 "person_attributes", "booth_sessions")


def delete_scope_data(db_path, booth_id=None, event_id=None, date=None, confirm_all=False):
    """Wipe every logged row (SCOPED_TABLES — interactions, presence
    sessions, tripwire crossings, etc.) matching booth_id/event_id/date —
    used by the Dashboard's single "clear data" action (booth-only,
    event-only, one-day-only, or any combination) and by the booth settings
    page's "reset this booth's data" action. Never touches registry/config
    tables (`booths`, `events`, `tripwires`) — only logged history.

    Destructive and irreversible; the caller is responsible for confirming
    with the user first. Requires at least one of booth_id/event_id/date,
    unless confirm_all=True (an explicit "yes, really wipe every scope"
    flag) — this is what stops an empty/default form submission from
    silently wiping the entire database."""
    if not booth_id and not event_id and not date and not confirm_all:
        raise ValueError("ต้องระบุ booth_id, event_id หรือ date อย่างน้อยหนึ่งอย่าง "
                          "(หรือส่ง confirm_all=True เพื่อล้างข้อมูลทั้งหมดโดยตั้งใจ)")
    where = []
    params: list = []
    if booth_id:
        where.append("booth_id = ?")
        params.append(booth_id)
    if event_id:
        where.append("event_id = ?")
        params.append(event_id)
    if date:
        start_ts, end_ts = _day_range_ts(date)
        where.append("ts >= ? AND ts < ?")
        params.extend([start_ts, end_ts])
    clause = " AND ".join(where) if where else "1=1"
    with _connect(db_path) as conn:
        for table in SCOPED_TABLES:
            conn.execute(f"DELETE FROM {table} WHERE {clause}", params)


def query_readiness_checks(db_path, event_id=None, booth_id=None, date=None, limit=100):
    clause, params = _scope_clause(event_id, booth_id, date)
    sql = "SELECT * FROM readiness_checks WHERE 1=1" + clause + " ORDER BY ts DESC LIMIT ?"
    with _connect(db_path) as conn:
        return _rows_as_dicts(conn.execute(sql, params + [limit]))


def query_heartbeats(db_path, event_id=None, booth_id=None, date=None, limit=500):
    clause, params = _scope_clause(event_id, booth_id, date)
    sql = "SELECT * FROM heartbeats WHERE 1=1" + clause + " ORDER BY ts DESC LIMIT ?"
    with _connect(db_path) as conn:
        return _rows_as_dicts(conn.execute(sql, params + [limit]))


def query_distinct(db_path, table, column):
    with _connect(db_path) as conn:
        try:
            rows = conn.execute(f"SELECT DISTINCT {column} FROM {table} ORDER BY {column}")
            return [r[0] for r in rows.fetchall()]
        except sqlite3.OperationalError:
            return []


def query_known_ids(db_path) -> dict:
    """Every booth_id / event_id that appears anywhere in the database — not
    just in interactions/health_events — so the Dashboard's filters and
    cleanup tools see booths/events that only ever produced, say, hold
    events or heartbeats."""
    booth_ids: set[str] = set()
    event_ids: set[str] = set()
    for table in SCOPED_TABLES:
        booth_ids.update(v for v in query_distinct(db_path, table, "booth_id") if v)
        event_ids.update(v for v in query_distinct(db_path, table, "event_id") if v)
    return {"booth_ids": sorted(booth_ids), "event_ids": sorted(event_ids)}


UNSET = object()  # distinguishes "leave event_id alone" from "set it to NULL" in update_booth()


def create_event(db_path, event_id: str, name: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("INSERT INTO events (id, name, created_ts) VALUES (?,?,?)",
                     (event_id, name, time.time()))


def ensure_event(db_path, event_id: str, name: str | None = None) -> None:
    """create_event, but a no-op if the id is already registered — for
    startup bootstrapping, where the same --event-id may be reused across
    multiple CLI launches (e.g. simulating several booths at one event)."""
    try:
        create_event(db_path, event_id, name or event_id)
    except sqlite3.IntegrityError:
        pass


def rename_event(db_path, event_id: str, name: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("UPDATE events SET name = ? WHERE id = ?", (name, event_id))


def delete_event(db_path, event_id: str) -> None:
    """Deleting an Event never deletes its member Booths — it's just a
    grouping label, so membership is dropped (booths become unassigned)
    before the event's own logged data is purged and its registry row
    removed."""
    with _connect(db_path) as conn:
        conn.execute("UPDATE booths SET event_id = NULL WHERE event_id = ?", (event_id,))
    delete_scope_data(db_path, event_id=event_id)
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM events WHERE id = ?", (event_id,))


def list_events(db_path) -> list[dict]:
    sql = """
        SELECT events.id, events.name, events.created_ts,
               (SELECT COUNT(*) FROM booths WHERE booths.event_id = events.id) AS booth_count
        FROM events ORDER BY events.name
    """
    with _connect(db_path) as conn:
        return _rows_as_dicts(conn.execute(sql))


def create_booth(db_path, booth_id: str, name: str, event_id: str | None = None,
                  open_time: str | None = None, close_time: str | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO booths (id, name, event_id, created_ts, open_time, close_time) "
            "VALUES (?,?,?,?,?,?)",
            (booth_id, name, event_id, time.time(), open_time, close_time),
        )


def update_booth(db_path, booth_id: str, name: str | None = None, event_id=UNSET,
                  open_time=UNSET, close_time=UNSET) -> None:
    sets = []
    params = []
    if name is not None:
        sets.append("name = ?")
        params.append(name)
    if event_id is not UNSET:
        sets.append("event_id = ?")
        params.append(event_id)  # None here means "unassign", written as NULL
    if open_time is not UNSET:
        sets.append("open_time = ?")
        params.append(open_time)  # None here means "no schedule", written as NULL
    if close_time is not UNSET:
        sets.append("close_time = ?")
        params.append(close_time)
    if not sets:
        return
    params.append(booth_id)
    with _connect(db_path) as conn:
        conn.execute(f"UPDATE booths SET {', '.join(sets)} WHERE id = ?", params)


def delete_booth(db_path, booth_id: str) -> None:
    delete_scope_data(db_path, booth_id=booth_id)
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM booths WHERE id = ?", (booth_id,))


def list_booths(db_path) -> list[dict]:
    sql = """
        SELECT booths.id, booths.name, booths.event_id, events.name AS event_name,
               booths.created_ts, booths.open_time, booths.close_time
        FROM booths LEFT JOIN events ON events.id = booths.event_id
        ORDER BY booths.name
    """
    with _connect(db_path) as conn:
        return _rows_as_dicts(conn.execute(sql))


def get_booth(db_path, booth_id: str) -> dict | None:
    with _connect(db_path) as conn:
        rows = _rows_as_dicts(conn.execute(
            "SELECT booths.id, booths.name, booths.event_id, events.name AS event_name, "
            "booths.open_time, booths.close_time "
            "FROM booths LEFT JOIN events ON events.id = booths.event_id WHERE booths.id = ?",
            (booth_id,),
        ))
    return rows[0] if rows else None


def count_booths(db_path) -> int:
    with _connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM booths").fetchone()[0]


def query_summary(db_path, event_id=None, booth_id=None, date=None):
    interactions = query_interactions(db_path, event_id, booth_id, date, limit=100000)
    health = query_health_events(db_path, event_id, booth_id, date, limit=100000)
    booths = {row["booth_id"] for row in interactions} | {row["booth_id"] for row in health}
    open_alerts = [h for h in health if h["status"] == "error"]
    return {
        "total_interactions": len(interactions),
        "unique_products": len({row["product_name"] for row in interactions}),
        "active_booths": len(booths),
        "open_alerts": len(open_alerts),
    }


# ------------------------------------------------------- Booth Session / Event

def query_event_sessions(db_path, event_id) -> list[dict]:
    """Every distinct (booth_id, calendar day) combination that has any
    logged activity under this event — the closest thing to a "Booth
    Session" (one row per booth-opening-instance) without a whole new
    always-has-to-be-kept-in-sync table: a session is derived on read from
    whichever days a booth actually produced data, rather than requiring
    every booth process to remember to open/close a session row of its own
    (the existing Booth Open/Close Schedule in web/booth_manager.py only
    starts/stops cameras — it never wrote a "session" record, so backfilling
    one from real event data is both simpler and always accurate for
    historical booths too). Returns newest-day-first
    [{"booth_id", "booth_name", "date", "open_time", "close_time"}] — one
    entry the Dashboard's "รวมทั้งงาน" (whole event) view can list and let the
    user drill into a single day/booth from."""
    tables = ("interactions", "health_events", "presence_sessions",
              "product_hold_events", "tripwire_crossings")
    union_sql = " UNION ".join(
        f"SELECT booth_id, date(ts,'unixepoch','localtime') AS d FROM {t} WHERE event_id = ?"
        for t in tables
    )
    sql = f"SELECT booth_id, d FROM ({union_sql}) GROUP BY booth_id, d ORDER BY d DESC"
    with _connect(db_path) as conn:
        rows = conn.execute(sql, [event_id] * len(tables)).fetchall()

    booths_by_id = {b["id"]: b for b in list_booths(db_path)}
    sessions = []
    for booth_id, d in rows:
        if not booth_id or not d:
            continue
        b = booths_by_id.get(booth_id, {})
        sessions.append({
            "booth_id": booth_id,
            "booth_name": b.get("name") or booth_id,  # booth may since have been deleted/renamed
            "date": d,
            "open_time": b.get("open_time"),
            "close_time": b.get("close_time"),
        })
    return sessions


def query_event_daily_breakdown(db_path, event_id, booth_id=None) -> list[dict]:
    """Per-day metrics for every day this event (optionally narrowed to one
    booth) has data for, oldest first — composes the existing single-day
    query functions per date rather than a fresh GROUP-BY-date SQL query, so
    every number here always matches what picking that exact day in the
    Dashboard's existing Date filter already shows (one source of truth,
    never a second aggregation path that could drift from the first)."""
    dates = sorted(query_available_dates(db_path, event_id, booth_id))
    breakdown = []
    for d in dates:
        summary = query_summary(db_path, event_id, booth_id, d)
        tripwire = query_tripwire_stats(db_path, event_id, booth_id, d)
        presence = query_presence_stats(db_path, event_id, booth_id, d)
        breakdown.append({
            "date": d,
            "total_interactions": summary["total_interactions"],
            "unique_products": summary["unique_products"],
            "visitors_in": tripwire["total_in"],
            "visitors_out": tripwire["total_out"],
            "peak_inside": tripwire["peak_inside"],
            "avg_dwell_sec": presence["avg_sec"],
            "presence_count": presence["count"],
        })
    return breakdown
