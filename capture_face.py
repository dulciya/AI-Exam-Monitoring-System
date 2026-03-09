import os
import cv2
import numpy as np
import face_recognition

os.makedirs("faces_db", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

student_id = input("Enter Student ID to capture face: ").strip()
img_path = f"faces_db/{student_id}.jpg"
emb_path = f"embeddings/{student_id}.npy"

cap = cv2.VideoCapture(0)
print("Align your face in the center. Press SPACE to capture, ESC to cancel.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("❌ Camera read failed.")
        break

    view = frame.copy()
    h, w = view.shape[:2]
    # draw a guide box
    cv2.rectangle(view, (w//4, h//4), (3*w//4, 3*h//4), (0,255,0), 2)

    cv2.imshow("Capture Face", view)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        print("Cancelled.")
        break
    elif key == 32:  # SPACE
        # detect largest face
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        if not boxes:
            print("No face detected. Try again (good light, face centered).")
            continue

        # pick largest (in case multiple)
        (top, right, bottom, left) = max(boxes, key=lambda b: (b[2]-b[0])*(b[1]-b[3]))
        face_crop = frame[top:bottom, left:right]
        if face_crop.size == 0:
            print("Bad crop, try again.")
            continue

        # save cropped face
        cv2.imwrite(img_path, face_crop)
        # compute and save encoding
        enc = face_recognition.face_encodings(rgb, [(top, right, bottom, left)])
        if not enc:
            print("Could not compute encoding. Try again.")
            continue
        np.save(emb_path, enc[0])

        print(f"✅ Saved face image: {img_path}")
        print(f"✅ Saved face encoding: {emb_path}")
        break

cap.release()
cv2.destroyAllWindows()
