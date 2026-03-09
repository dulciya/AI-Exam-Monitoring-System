import os
import cv2
import numpy as np
import sqlite3
import face_recognition

DB = "exam_guard.db"
EMB_DIR = "embeddings"

# Load known encodings
known_encodings = []
known_ids = []

for file in os.listdir(EMB_DIR):
    if file.endswith(".npy"):
        sid = os.path.splitext(file)[0]
        enc = np.load(os.path.join(EMB_DIR, file))
        known_encodings.append(enc)
        known_ids.append(sid)

print(f"Loaded {len(known_ids)} registered encodings.")

cap = cv2.VideoCapture(0)
TOLERANCE = 0.5  # lower=stricter

while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encs  = face_recognition.face_encodings(rgb, boxes)

    for (top, right, bottom, left), face_enc in zip(boxes, encs):
        name_display = "Unknown"
        student_id = None

        if known_encodings:
            # Find best match
            dists = face_recognition.face_distance(known_encodings, face_enc)
            idx = int(np.argmin(dists))
            if dists[idx] <= TOLERANCE:
                student_id = known_ids[idx]
                with sqlite3.connect(DB) as conn:
                    c = conn.cursor()
                    c.execute("SELECT name, seat FROM students WHERE student_id=?", (student_id,))
                    row = c.fetchone()
                    if row:
                        name_display = f"{row[0]} - Seat {row[1]}"

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0) if student_id else (0,0,255), 2)
        cv2.putText(frame, name_display, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Seat Verification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
