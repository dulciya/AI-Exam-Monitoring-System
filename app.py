import os, time, glob, sqlite3, threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import face_recognition
from flask import Flask, render_template, Response, jsonify

# -------------------- CONFIG --------------------
DB_PATH = "exam_guard.db"
EMB_DIR = "embeddings"           # where capture_face.py saved .npy files
CAM_INDEX = 0                    # change to 1 if Camo grabs 0
TOLERANCE = 0.50                 # lower = stricter
PROCESS_EVERY_N = 2              # run recognition every N frames
SMOOTH_N = 5                     # majority vote window
# ------------------------------------------------

app = Flask(__name__)

# Global state shared with the stream
LOCK = threading.Lock()
STATUS = {
    "state": "unknown",          # 'unknown' | 'verified' | 'mismatch'
    "student_id": None,
    "name": None,
    "seat": None,
    "distance": None,
    "last_ts": None
}

known_ids = []       # list[str]
known_encs = []      # list[np.array]
id_to_name = {}      # SID -> Name
id_to_seat = {}      # SID -> Seat

def db_connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def load_known_faces():
    """Load embeddings/*.npy and map them to students in DB."""
    global known_ids, known_encs, id_to_name, id_to_seat
    known_ids, known_encs = [], []
    id_to_name, id_to_seat = {}, {}

    # read students for names/seats
    with db_connect() as conn:
        c = conn.cursor()
        c.execute("SELECT student_id, name, seat FROM students")
        for sid, name, seat in c.fetchall():
            id_to_name[sid] = name
            id_to_seat[sid] = seat

    # load embeddings (*.npy where filename is <SID>.npy)
    for f in glob.glob(os.path.join(EMB_DIR, "*.npy")):
        sid = os.path.splitext(os.path.basename(f))[0]
        try:
            enc = np.load(f)
            if enc.ndim == 1:  # single vector saved
                known_ids.append(sid)
                known_encs.append(enc)
        except Exception:
            pass

def ensure_entry_logged(sid):
    """Log to entry_log once per session if not already present."""
    with db_connect() as conn:
        c = conn.cursor()
        c.execute("SELECT 1 FROM entry_log WHERE student_id=? ORDER BY entry_time DESC LIMIT 1", (sid,))
        already = c.fetchone()
        if not already:
            name = id_to_name.get(sid, "")
            seat = id_to_seat.get(sid, "")
            now = datetime.now().isoformat(timespec="seconds")
            c.execute(
                "INSERT INTO entry_log (student_id, name, seat, entry_time) VALUES (?,?,?,?)",
                (sid, name, seat, now)
            )
            conn.commit()

def set_status(state, sid=None, dist=None):
    with LOCK:
        STATUS["state"] = state
        STATUS["student_id"] = sid
        STATUS["name"] = id_to_name.get(sid) if sid else None
        STATUS["seat"] = id_to_seat.get(sid) if sid else None
        STATUS["distance"] = None if dist is None else float(dist)
        STATUS["last_ts"] = datetime.now().strftime("%H:%M:%S")

def gen_frames():
    """Video stream generator with inline face verification."""
    load_known_faces()
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        # try another camera index automatically
        for idx in [1, 2]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                break

    last_preds = deque(maxlen=SMOOTH_N)
    frame_i = 0

    set_status("unknown", None, None)

    while True:
        ok, frame = cap.read()
        if not ok:
            # small pause then continue trying
            time.sleep(0.05)
            continue

        # downscale for faster recognition
        small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        label_text = "Unknown / No face"
        color = (0, 0, 255)   # red

        if frame_i % PROCESS_EVERY_N == 0:
            face_locs = face_recognition.face_locations(rgb_small, model="hog")
            encs = face_recognition.face_encodings(rgb_small, face_locs)

            pred_sid = None
            pred_dist = None

            if encs:
                # take first face (single-person station)
                e = encs[0]
                if known_encs:
                    dists = face_recognition.face_distance(known_encs, e)
                    j = int(np.argmin(dists))
                    best_dist = dists[j]
                    if best_dist < TOLERANCE:
                        pred_sid, pred_dist = known_ids[j], best_dist

            # smooth predictions
            last_preds.append(pred_sid)
            # majority vote
            if last_preds and any(p is not None for p in last_preds):
                # choose most common non-None
                vals = [p for p in last_preds if p is not None]
                winner = max(set(vals), key=vals.count)
                pred_sid = winner
            else:
                pred_sid = None

            if pred_sid:
                # verified
                set_status("verified", pred_sid, pred_dist)
                ensure_entry_logged(pred_sid)
            else:
                # unknown/mismatch
                # If a face exists but not matched, call "mismatch"
                state = "mismatch" if encs else "unknown"
                set_status(state, None, None)

        # Overlay status on the frame
        with LOCK:
            st = STATUS.copy()

        if st["state"] == "verified":
            label_text = f"Verified: {st['name']}  | Seat {st['seat']}"
            color = (0, 200, 0)
        elif st["state"] == "mismatch":
            label_text = "ALERT — FACE MISMATCH"
            color = (0, 0, 255)
        else:
            label_text = "Unknown / No face"
            color = (0, 0, 255)

        # draw banner
        cv2.rectangle(frame, (0,0), (frame.shape[1], 40), color, -1)
        cv2.putText(frame, label_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # jpeg encode
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

        frame_i += 1

# -------------------- ROUTES --------------------
@app.route("/")
def index():
    # main page with alert banner + video feed
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/status")
def api_status():
    with LOCK:
        return jsonify(STATUS)

# optional: simple seatmap page (reuses your earlier logic)
def get_seat_status():
    seats = {}
    with db_connect() as conn:
        c = conn.cursor()
        c.execute("SELECT student_id, name, seat FROM students")
        for sid, name, seat in c.fetchall():
            seats[seat] = {"student_id": sid, "name": name, "status": "empty"}
        c.execute("SELECT student_id, seat FROM entry_log")
        for sid, seat in c.fetchall():
            if seat in seats:
                seats[seat]["status"] = "present"
    return seats

@app.route("/seatmap")
def seatmap():
    seats = get_seat_status()
    # quick inline HTML to avoid a second template for now
    html = ["<h2 style='font-family:sans-serif'>Seat Map</h2><div style='display:grid;grid-template-columns:repeat(5,120px);gap:10px'>"]
    for seat, info in seats.items():
        bg = "#2e7d32" if info["status"] == "present" else "#7a7a7a"
        html.append(f"<div style='background:{bg};color:#fff;border-radius:10px;height:100px;display:flex;flex-direction:column;align-items:center;justify-content:center;font-family:sans-serif'><div>{seat}</div><div style='font-size:12px'>{info['name']}</div></div>")
    html.append("</div><p style='font-family:sans-serif'><a href=\"/\">Back</a></p>")
    return "\n".join(html)

def get_seat_status():
    conn = sqlite3.connect("exam_guard.db")
    c = conn.cursor()
    c.execute("SELECT student_id, seat FROM students")
    students = c.fetchall()

    c.execute("SELECT student_id FROM entry_log")
    logged_in = {row[0] for row in c.fetchall()}
    conn.close()

    seat_status = []
    for sid, seat in students:
        status = "green" if sid in logged_in else "grey"
        seat_status.append({"seat": seat, "status": status})
    return seat_status

@app.route("/seats")
def seats():
    return jsonify(get_seat_status())


# ------------------------------------------------
if __name__ == "__main__":
    load_known_faces()
    app.run(debug=True)
