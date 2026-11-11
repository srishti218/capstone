"""SQLite persistence for study jobs, model output and review events."""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY, metadata TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS frames (
    result_id TEXT NOT NULL REFERENCES results(id), position INTEGER NOT NULL,
    image BLOB NOT NULL, overlay BLOB NOT NULL, PRIMARY KEY(result_id, position)
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY, result_id TEXT NOT NULL REFERENCES results(id),
    request_id TEXT UNIQUE NOT NULL, actor TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('approved', 'rejected')),
    note TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS decisions_result ON decisions(result_id, id);
CREATE TRIGGER IF NOT EXISTS decisions_no_update BEFORE UPDATE ON decisions
BEGIN SELECT RAISE(ABORT, 'decisions are append-only'); END;
CREATE TRIGGER IF NOT EXISTS decisions_no_delete BEFORE DELETE ON decisions
BEGIN SELECT RAISE(ABORT, 'decisions are append-only'); END;
CREATE TABLE IF NOT EXISTS state (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (
    study TEXT PRIMARY KEY, seq INTEGER NOT NULL, status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0, next_at INTEGER NOT NULL DEFAULT 0
);
"""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path=None):
        self.path = Path(path or os.getenv("RADREVIEW_DB", "runtime/review.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def has_result(self, result_id):
        with self.connect() as db:
            return db.execute("SELECT 1 FROM results WHERE id=?", (result_id,)).fetchone() is not None

    def save_result(self, result_id, metadata, frames):
        with self.connect() as db:
            row = db.execute(
                "INSERT OR IGNORE INTO results VALUES (?, ?, ?)",
                (result_id, json.dumps(metadata, allow_nan=False), utc_now()),
            )
            if row.rowcount:
                db.executemany(
                    "INSERT INTO frames VALUES (?, ?, ?, ?)",
                    ((result_id, i, image, overlay) for i, (image, overlay) in enumerate(frames)),
                )

    def results(self):
        with self.connect() as db:
            rows = db.execute("""
                SELECT r.*, COALESCE((SELECT decision FROM decisions d
                    WHERE d.result_id=r.id ORDER BY d.id DESC LIMIT 1), 'pending') AS status
                FROM results r ORDER BY r.created_at DESC, r.id
            """)
            return [
                dict(json.loads(row["metadata"]), id=row["id"],
                     created_at=row["created_at"], status=row["status"])
                for row in rows
            ]

    def frame(self, result_id, position, layer):
        with self.connect() as db:
            row = db.execute(
                "SELECT image, overlay FROM frames WHERE result_id=? AND position=?",
                (result_id, position),
            ).fetchone()
            return row[layer] if row else None

    def audit(self, result_id=None):
        with self.connect() as db:
            if result_id is None:
                rows = db.execute("SELECT * FROM decisions ORDER BY id DESC")
            else:
                rows = db.execute(
                    "SELECT * FROM decisions WHERE result_id=? ORDER BY id DESC", (result_id,)
                )
            return [dict(row) for row in rows]

    def review(self, result_id, request_id, actor, decision, note):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            values = (result_id, request_id, actor, decision, note)
            existing = db.execute(
                "SELECT * FROM decisions WHERE request_id=?", (request_id,)
            ).fetchone()
            if existing:
                keys = ("result_id", "request_id", "actor", "decision", "note")
                if tuple(existing[key] for key in keys) != values:
                    raise ValueError("This request ID already belongs to another decision.")
                return dict(existing)
            entry = db.execute(
                """INSERT INTO decisions (result_id, request_id, actor, decision, note, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (*values, utc_now()),
            )
            return dict(db.execute("SELECT * FROM decisions WHERE id=?", (entry.lastrowid,)).fetchone())

    def cursor(self):
        with self.connect() as db:
            row = db.execute("SELECT value FROM state WHERE name='orthanc_cursor'").fetchone()
            return row["value"] if row else 0

    def enqueue(self, changes, last):
        with self.connect() as db:
            db.executemany("""
                INSERT INTO jobs(study, seq, status) VALUES (?, ?, 'queued')
                ON CONFLICT(study) DO UPDATE SET seq=excluded.seq, status='queued',
                attempts=0, next_at=0 WHERE excluded.seq > jobs.seq
            """, changes)
            db.execute("""
                INSERT INTO state VALUES ('orthanc_cursor', ?)
                ON CONFLICT(name) DO UPDATE SET value=max(state.value, excluded.value)
            """, (last,))

    def due_jobs(self, now):
        with self.connect() as db:
            return db.execute(
                """SELECT * FROM jobs WHERE status IN ('queued', 'retry', 'awaiting weights')
                   AND next_at<=? ORDER BY seq LIMIT 8""", (now,)
            ).fetchall()

    def update_job(self, job, status, next_at=0):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status=?, next_at=?, attempts=attempts+1 WHERE study=? AND seq=?",
                (status, next_at, job["study"], job["seq"]),
            )

    def job_counts(self):
        with self.connect() as db:
            return dict(db.execute("SELECT status, count(*) FROM jobs GROUP BY status"))
