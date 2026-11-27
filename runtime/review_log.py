"""Durable review history (approve / reject decisions) in a local SQLite file.

Every review is written here before it becomes visible, together with a snapshot of the payment
it decided (parties, amount, score, automated decision). The snapshot keeps the history readable
after a restart even for simulated requests, which live only in process memory.

If the file cannot be opened or written (for example a read-only filesystem), the log switches to
an in-memory SQLite database, logs a warning, and keeps working; `persisted` then reports False.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Any

log = logging.getLogger("secflow.review_log")

SCHEMA = """
CREATE TABLE IF NOT EXISTS review_history (
    review_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    request_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('approve', 'reject')),
    reason TEXT NOT NULL,
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL,
    simulated_time INTEGER NOT NULL,
    original_decision TEXT NOT NULL,
    original_decision_label TEXT NOT NULL,
    original_risk_score REAL,
    resulting_outcome TEXT NOT NULL,
    origin TEXT NOT NULL,
    event_time TEXT NOT NULL,
    sender_id TEXT NOT NULL,
    sender_name TEXT NOT NULL,
    recipient_id TEXT NOT NULL,
    recipient_name TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS review_history_seq ON review_history (seq);
CREATE INDEX IF NOT EXISTS review_history_request ON review_history (request_id);
"""


@dataclass(frozen=True)
class ReviewHistoryEntry:
    review_id: str
    seq: int
    request_id: str
    action: str
    reason: str
    actor: str
    created_at: str
    simulated_time: int
    original_decision: str
    original_decision_label: str
    original_risk_score: float | None
    resulting_outcome: str
    origin: str
    event_time: str
    sender_id: str
    sender_name: str
    recipient_id: str
    recipient_name: str
    amount: float
    currency: str


COLUMNS = tuple(f.name for f in fields(ReviewHistoryEntry))
_SELECT = f"SELECT {', '.join(COLUMNS)} FROM review_history"
_INSERT = f"INSERT INTO review_history ({', '.join(COLUMNS)}) VALUES ({', '.join('?' for _ in COLUMNS)})"


def _memory_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
    db.executescript(SCHEMA)
    return db


class ReviewLog:
    def __init__(self, directory: Path | None) -> None:
        self._lock = threading.RLock()
        self.path: Path | None = None
        self.persisted = False
        self.error: str | None = None
        self._db = self._open(directory)

    def _open(self, directory: Path | None) -> sqlite3.Connection:
        if directory is None:
            return _memory_db()
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "review_history.sqlite3"
            db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
        except (OSError, sqlite3.Error) as error:
            self.error = f"Review history file unavailable ({error}); keeping it in memory."
            log.warning(self.error)
            return _memory_db()
        self.path, self.persisted = path, True
        return db

    def _fallback_to_memory(self, error: Exception) -> None:
        """Copy what is still readable into memory so the history keeps working after a write failure."""
        self.error = f"Review history write failed ({error}); continuing in memory."
        log.warning(self.error)
        try:
            rows = self._db.execute(_SELECT).fetchall()
        except sqlite3.Error:
            rows = []
        memory = _memory_db()
        memory.executemany(_INSERT, rows)
        self._db, self.persisted = memory, False

    # ------------------------------------------------------------------
    def next_seq(self) -> int:
        with self._lock:
            (value,) = self._db.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM review_history").fetchone()
            return int(value)

    def append(self, entry: ReviewHistoryEntry) -> None:
        row = astuple(entry)
        with self._lock:
            try:
                self._db.execute(_INSERT, row)
            except sqlite3.IntegrityError:
                raise
            except sqlite3.Error as error:
                self._fallback_to_memory(error)
                self._db.execute(_INSERT, row)

    def entries(self) -> list[ReviewHistoryEntry]:
        with self._lock:
            rows = self._db.execute(f"{_SELECT} ORDER BY seq").fetchall()
        return [ReviewHistoryEntry(*row) for row in rows]

    def page(
        self, action: str | None, q: str | None, page: int, page_size: int, sort: str = "newest"
    ) -> tuple[list[ReviewHistoryEntry], int, dict[str, int]]:
        where: list[str] = []
        params: list[Any] = []
        if action:
            where.append("action = ?")
            params.append(action)
        query = (q or "").strip().lower()
        if query:
            searchable = ("request_id", "sender_name", "recipient_name", "sender_id", "recipient_id", "actor", "reason")
            where.append("(" + " OR ".join(f"lower({c}) LIKE ?" for c in searchable) + ")")
            params += [f"%{query}%"] * len(searchable)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        order = "seq ASC" if sort == "oldest" else "seq DESC"
        with self._lock:
            (total,) = self._db.execute(f"SELECT COUNT(*) FROM review_history {clause}", params).fetchone()
            rows = self._db.execute(
                f"{_SELECT} {clause} ORDER BY {order} LIMIT ? OFFSET ?", [*params, page_size, (page - 1) * page_size]
            ).fetchall()
            counts = dict(self._db.execute("SELECT action, COUNT(*) FROM review_history GROUP BY action").fetchall())
        totals = {"approve": int(counts.get("approve", 0)), "reject": int(counts.get("reject", 0))}
        return [ReviewHistoryEntry(*row) for row in rows], int(total), totals

    def status(self) -> dict[str, Any]:
        return {"persisted": self.persisted, "path": str(self.path) if self.persisted else None, "error": self.error}

    def close(self) -> None:
        with self._lock:
            self._db.close()
