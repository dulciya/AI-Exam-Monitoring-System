import sqlite3
from pathlib import Path

DB_PATH = Path("exam_guard.db")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()

        # Students table: supports fingerprint (template_pos) and basic identity
        c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id   TEXT PRIMARY KEY,
            name         TEXT NOT NULL,
            seat         TEXT NOT NULL,
            template_pos INTEGER,            -- fingerprint template index (nullable)
            enrolled_at  TEXT NOT NULL
        )
        """)

        # Entry log: includes optional fingerprint match info
        c.execute("""
        CREATE TABLE IF NOT EXISTS entry_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id  TEXT NOT NULL,
            name        TEXT NOT NULL,
            seat        TEXT NOT NULL,
            matched_pos INTEGER,             -- fingerprint template index if used
            accuracy    REAL,                -- fingerprint score if used
            entry_time  TEXT NOT NULL
        )
        """)

        # Events (malpractice) — we’ll use this in monitoring phase
        c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            seat       TEXT,
            event_type TEXT,                 -- e.g., 'PEEK_LEFT','PHONE','EMPTY_SEAT'
            confidence REAL,
            timestamp  TEXT
        )
        """)
        conn.commit()
    print(f"✅ Database initialized at {DB_PATH.resolve()}")

if __name__ == "__main__":
    init_db()
