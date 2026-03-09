# monitor.py
import cv2
import mediapipe as mp
import numpy as np
import time
import base64
import requests
from collections import defaultdict, deque
from math import degrees, atan2

# ---- CONFIG ----
DASHBOARD_URL = "http://localhost:5000/alert"  # Flask dashboard endpoint
VIDEO_SOURCE = 0  # 0 = default webcam
FPS = 10
HEAD_YAW_THRESHOLD_DEG = 25
LOOK_AWAY_FRAMES_TO_ALERT = int(2 * FPS)
SUSPICION_SCORE_THRESHOLD = 3.0
MAX_HISTORY = 5 * FPS
SEAT_ZONES = [
    {"id": "A1", "poly": [(50,400),(200,400),(200,550),(50,550)]},
    {"id": "A2", "poly": [(220,400),(370,400),(370,550),(220,550)]},
]

# ---- helpers ----
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face.FaceMesh(max_num_faces=10, refine_landmarks=False)
hands = mp_hands.Hands(max_num_hands=4)

LANDMARK_IDS = [33, 263, 1, 61, 291, 199]
MODEL_POINTS_3D = np.array([
    (-30.0, -50.0, -30.0),
    (30.0, -50.0, -30.0),
    (0.0, 0.0, 0.0),
    (-25.0, 40.0, -30.0),
    (25.0, 40.0, -30.0),
    (0.0, 90.0, -50.0),
], dtype=np.float32)

def euler_from_R(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = atan2(R[2,1], R[2,2])
        y = atan2(-R[2,0], sy)
        z = atan2(R[1,0], R[0,0])
    else:
        x = atan2(-R[1,2], R[1,1])
        y = atan2(-R[2,0], sy)
        z = 0
    return degrees(z), degrees(y), degrees(x)

def get_seat_id(point):
    x, y = int(point[0]), int(point[1])
    for z in SEAT_ZONES:
        poly = np.array(z["poly"], dtype=np.int32)
        if cv2.pointPolygonTest(poly, (x,y), False) >= 0:
            return z["id"]
    return None

def frame_to_base64(img):
    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return base64.b64encode(buf).decode('ascii')

# ---- main loop ----
def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    ret, frame = cap.read()
    if not ret:
        print("Cannot open camera.")
        return
    h, w = frame.shape[:2]
    camera_matrix = np.array([[w,0,w//2],[0,w,h//2],[0,0,1]], dtype=np.float32)
    dist_coeffs = np.zeros((4,1))

    suspicion = defaultdict(float)
    look_away_counts = defaultdict(int)
    history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(image_rgb)
        hand_results = hands.process(image_rgb)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                h_lm = [(int(face_landmarks.landmark[idx].x*w), int(face_landmarks.landmark[idx].y*h)) for idx in LANDMARK_IDS]
                image_points = np.array(h_lm, dtype=np.float32)
                ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs)
                if ok:
                    R, _ = cv2.Rodrigues(rvec)
                    yaw, pitch, roll = euler_from_R(R)
                else:
                    yaw, pitch, roll = 0.0, 0.0, 0.0

                cx, cy = int(np.mean([p[0] for p in h_lm])), int(np.mean([p[1] for p in h_lm]))
                seat = get_seat_id((cx, cy))
                if seat:
                    cv2.putText(frame, f"{seat}", (cx-20, cy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                if abs(yaw) > HEAD_YAW_THRESHOLD_DEG and seat:
                    suspicion[seat] += 1.0
                if seat and suspicion[seat] >= SUSPICION_SCORE_THRESHOLD:
                    payload = {"seat": seat, "reason": "head_away", "score": float(suspicion[seat]), "yaw": float(yaw), "pitch": float(pitch), "frame_b64": frame_to_base64(frame), "timestamp": time.time()}
                    try: requests.post(DASHBOARD_URL, json=payload, timeout=1.5)
                    except: pass
                    suspicion[seat] = 0.0

        if hand_results.multi_hand_landmarks:
            centers = []
            for hld in hand_results.multi_hand_landmarks:
                xs = [lm.x*w for lm in hld.landmark]
                ys = [lm.y*h for lm in hld.landmark]
                centers.append((int(np.mean(xs)), int(np.mean(ys))))
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    d = np.hypot(centers[i][0]-centers[j][0], centers[i][1]-centers[j][1])
                    if d < 80:
                        seat_i = get_seat_id(centers[i])
                        seat_j = get_seat_id(centers[j])
                        for s in (seat_i, seat_j):
                            if s:
                                payload = {"seat": s, "reason": "hand_proximity", "score": float(suspicion[s]), "frame_b64": frame_to_base64(frame), "timestamp": time.time()}
                                try: requests.post(DASHBOARD_URL, json=payload, timeout=1.5)
                                except: pass

        cv2.imshow("Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elapsed = time.time() - start
        time.sleep(max(0, 1.0/FPS - elapsed))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
