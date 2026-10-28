"""
Database module for the Adaptive AI Learning System.

Uses SQLite (built into Python) to store student logins and quiz attempts.
The database file is created automatically on first run.
"""

from __future__ import annotations

import hashlib
import pathlib
import sqlite3
from datetime import datetime, timezone

# ------------------------------------------------------------------
# The default password for every student account.
# Change this constant to change the password for all students.
# ------------------------------------------------------------------
DEFAULT_PASSWORD = "student123"

# Path to the SQLite database file
DB_PATH = pathlib.Path("data") / "app.db"

# Special student names for specific IDs (all others are "Student <id>")
SPECIAL_NAMES: dict[int, str] = {
    47: "Siddhi Nikam",
    49: "Disha Patel",
    51: "Akshay Patil",
    58: "Srushti Shinde",
}


def _hash_password(plain_text: str) -> str:
    """
    Convert a plain-text password into a SHA-256 hash string.
    We never store the real password, only this hash.
    """
    return hashlib.sha256(plain_text.encode()).hexdigest()


def get_connection() -> sqlite3.Connection:
    """
    Open (or create) the SQLite database and return a connection object.
    Also turns on foreign keys for data integrity.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dictionaries
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialise_db() -> None:
    """
    Create the database tables if they do not already exist,
    and seed the students table with IDs 1 to 60.
    """
    conn = get_connection()
    with conn:
        # Table for student login credentials
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id   INTEGER PRIMARY KEY,
                name         TEXT    NOT NULL,
                password     TEXT    NOT NULL
            )
        """)

        # Table for every quiz answer the student submits
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  INTEGER NOT NULL,
                question_id INTEGER NOT NULL,
                topic       TEXT    NOT NULL,
                difficulty  TEXT    NOT NULL,
                correct     INTEGER NOT NULL,   -- 1 for correct, 0 for wrong
                time_taken  REAL    NOT NULL,   -- seconds
                timestamp   TEXT    NOT NULL
            )
        """)

        # Seed student accounts only if the table is empty
        count = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        if count == 0:
            hashed = _hash_password(DEFAULT_PASSWORD)
            for sid in range(1, 61):
                name = SPECIAL_NAMES.get(sid, f"Student {sid}")
                conn.execute(
                    "INSERT INTO students (student_id, name, password) VALUES (?, ?, ?)",
                    (sid, name, hashed),
                )
    conn.close()


def check_login(student_id: int, password: str) -> dict | None:
    """
    Check if the student ID and password are correct.

    Returns a dict with student_id and name if correct, or None if wrong.
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT student_id, name, password FROM students WHERE student_id = ?",
        (student_id,),
    ).fetchone()
    conn.close()

    if row is None:
        return None   # Student ID not found

    if row["password"] != _hash_password(password):
        return None   # Password is wrong

    return {"student_id": row["student_id"], "name": row["name"]}


def save_attempt(
    student_id: int,
    question_id: int,
    topic: str,
    difficulty: str,
    correct: bool,
    time_taken: float,
) -> None:
    """
    Save one quiz answer to the database.
    correct=True means the student got it right.
    """
    conn = get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO attempts
                (student_id, question_id, topic, difficulty, correct, time_taken, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                question_id,
                topic,
                difficulty,
                int(correct),
                float(time_taken),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    conn.close()


def get_attempts(student_id: int) -> list[dict]:
    """
    Return all quiz attempts for one student as a list of dictionaries.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM attempts WHERE student_id = ? ORDER BY id",
        (student_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
