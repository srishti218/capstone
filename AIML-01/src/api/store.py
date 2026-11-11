"""Durable local scores, alert cases and analyst labels."""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def state_dir():
    return Path(os.getenv("FRAUD_STATE_DIR", "artifacts/state"))


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def database():
    path = state_dir() / "cases.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY, event TEXT NOT NULL, score TEXT NOT NULL,
                    created_at TEXT NOT NULL, status TEXT, label TEXT, updated_at TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS case_queue ON transactions(status, created_at)")
            yield conn
    finally:
        conn.close()


def event_json(payload):
    return json.dumps({key: value for key, value in payload.items() if value is not None},
                      sort_keys=True, allow_nan=False, separators=(",", ":"))


def existing_score(transaction_id, payload):
    with database() as conn:
        row = conn.execute("SELECT event, score FROM transactions WHERE id = ?", (transaction_id,)).fetchone()
    return read_score(row, payload) if row else None


def read_score(row, payload):
    if event_json(json.loads(row["event"])) != event_json(payload):
        raise ValueError("Transaction ID already belongs to a different event.")
    return json.loads(row["score"])


def save_score(transaction_id, payload, score):
    with database() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO transactions (id, event, score, created_at, status)
            VALUES (?, ?, ?, ?, ?)
        """, (transaction_id, event_json(payload), json.dumps(score), now(),
              "open" if score["decision"] == "REVIEW" else None))
        row = conn.execute("SELECT event, score FROM transactions WHERE id = ?", (transaction_id,)).fetchone()
        return read_score(row, payload)


def case_dict(row):
    return {**json.loads(row["score"]), **{key: row[key] for key in ("status", "label", "created_at", "updated_at")}}


def cases(status, limit):
    with database() as conn:
        counts = {row["status"]: row["total"] for row in conn.execute(
            "SELECT status, count(*) AS total FROM transactions WHERE status IS NOT NULL GROUP BY status"
        )}
        rows = conn.execute("""
            SELECT * FROM transactions WHERE status IS NOT NULL AND (? IS NULL OR status = ?)
            ORDER BY created_at DESC, id DESC LIMIT ?
        """, (status, status, limit)).fetchall()
    return {"counts": counts, "cases": [case_dict(row) for row in rows]}


def update_case(transaction_id, status, label, label_provided):
    with database() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM transactions WHERE id = ? AND status IS NOT NULL", (transaction_id,)).fetchone()
        if row is None:
            return None
        status = status or row["status"]
        label = label if label_provided else row["label"]
        if status == "closed" and label is None:
            raise ValueError("A closed case requires a feedback label.")
        if (status, label) != (row["status"], row["label"]):
            conn.execute("UPDATE transactions SET status = ?, label = ?, updated_at = ? WHERE id = ?",
                         (status, label, now(), transaction_id))
            row = conn.execute("SELECT * FROM transactions WHERE id = ?", (transaction_id,)).fetchone()
    return case_dict(row)


def monitoring_rows(limit):
    with database() as conn:
        recent = conn.execute("SELECT event FROM transactions ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
        labeled = conn.execute("""
            SELECT id, event, label, updated_at, score FROM transactions WHERE status = 'closed'
            ORDER BY updated_at DESC, id DESC LIMIT ?
        """, (limit,)).fetchall()
    return [json.loads(row["event"]) for row in recent], [
        (row["id"], json.loads(row["event"]), row["label"], row["updated_at"],
         json.loads(row["score"]).get("card_history")) for row in labeled
    ]
