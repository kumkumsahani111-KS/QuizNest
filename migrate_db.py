
import sqlite3
from pathlib import Path

db_path = Path(__file__).parent / "instance" / "quiznest.db"

if not db_path.exists():
    raise FileNotFoundError("quiznest.db nahi mili!")

with sqlite3.connect(db_path) as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY,
            name VARCHAR(120) NOT NULL,
            username VARCHAR(80) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            created_at DATETIME
        )
    """)

    columns = [
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(teacher_quizzes)"
        )
    ]

    if not columns:
        raise RuntimeError("teacher_quizzes table nahi mili!")

    if "teacher_id" not in columns:
        conn.execute("""
            ALTER TABLE teacher_quizzes
            ADD COLUMN teacher_id INTEGER
            REFERENCES teachers(id)
        """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS
        ix_teacher_quizzes_teacher_id
        ON teacher_quizzes (teacher_id)
    """)

    conn.commit()

print("Database migration completed successfully!")