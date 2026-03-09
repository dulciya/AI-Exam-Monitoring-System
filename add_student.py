import sqlite3
from datetime import datetime

DB = "exam_guard.db"

def add_or_update_student(student_id: str, name: str, seat: str):
    enrolled_at = datetime.now().isoformat(timespec='seconds')
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        # Ensure row exists; keep existing template_pos if present
        c.execute("""
            INSERT INTO students (student_id, name, seat, template_pos, enrolled_at)
            VALUES (?, ?, ?, NULL, ?)
            ON CONFLICT(student_id) DO UPDATE SET
                name=excluded.name,
                seat=excluded.seat,
                enrolled_at=excluded.enrolled_at
        """, (student_id, name, seat, enrolled_at))
        conn.commit()
    print(f"✅ Added/Updated: {name} ({student_id}) → Seat {seat}")

if __name__ == "__main__":
    while True:
        sid = input("Enter Student ID (or 'q' to quit): ").strip()
        if sid.lower() == 'q':
            break
        name = input("Enter Student Name: ").strip()
        seat = input("Enter Seat Number (e.g., A12): ").strip()
        add_or_update_student(sid, name, seat)