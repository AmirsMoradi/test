import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import insightface
import time
from collections import deque

# ==== Basic configuration ====
RTSP_URL = "rtsp://admin:Qwerty123@10.41.41.11:554/cam/realmonitor?channel=1&subtype=0"
DB_PATH = "db/faces.db"

RESIZE_WIDTH = 640
FRAME_SKIP = 2
SHOW_WINDOW = True

UPDATE_INTERVAL = 0.5
HISTORY_SECONDS = 1.2
MOTION_SCALE = 8000.0

FINAL_SCORE_THRESHOLD = 70.0    # Liveness threshold
SIMILARITY_THRESHOLD = 0.5     # Face match threshold

# ==== Mediapipe setup ====
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# ==== InsightFace setup ====
print("⏳ Loading InsightFace...")
app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)
print("✅ Model loaded.\n")

# ==== Load embeddings from SQLite ====
print("📂 Loading embeddings...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
persons = {row[0]: row[1]
           for row in cursor.execute("SELECT id, name FROM persons")}
embeddings = []
person_ids = []


def safe_normalize(v):
    v = np.array(v, dtype=np.float32, copy=True)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


for pid, emb_blob in cursor.execute("SELECT person_id, embedding FROM embeddings"):
    emb = np.frombuffer(emb_blob, dtype=np.float32).copy()
    emb = safe_normalize(emb)
    embeddings.append(emb)
    person_ids.append(pid)

embeddings = np.array(embeddings)
print(f"✅ Loaded {len(embeddings)} embeddings.\n")
conn.close()


def find_best_match(new_emb, embeddings, person_ids):
    if len(embeddings) == 0:
        return None, 0.0
    new_emb = safe_normalize(new_emb)
    sims = np.dot(embeddings, new_emb)
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    best_person = persons.get(person_ids[best_idx], "Unknown")
    return best_person, best_score

# ==== Liveness functions ====


def eye_aspect_ratio(eye_pts):
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def head_pose_variation(centroids):
    if len(centroids) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(centroids)):
        total += np.linalg.norm(centroids[i] - centroids[i - 1])
    return total / (len(centroids) - 1)


def compute_texture_real_score(face_gray):
    face_gray = cv2.resize(face_gray, (128, 128))
    lap = cv2.Laplacian(face_gray, cv2.CV_64F)
    var = lap.var()
    norm = (var - 40.0) / 120.0
    norm = np.clip(norm, 0.0, 1.0)
    return float(norm * 100.0)


# ==== Video capture ====
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open RTSP stream")

frame_count = 0
motion_history = deque()
blink_timestamps = deque()
last_update_time = time.time()
liveness_score = 0.0
realness_score = 0.0
final_score = 0.0

LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
BLINK_THRESHOLD = 0.2

print("✅ Liveness + Recognition pipeline started...")

with mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
) as face_mesh:

    prev_eye_open = True
    recognition_active = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ No frame from RTSP.")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / float(w)
        frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        now = time.time()
        face_found = False
        face_roi = None

        if results.multi_face_landmarks:
            face_found = True
            face_landmarks = results.multi_face_landmarks[0]
            pts = np.array([(lm.x * w, lm.y * h)
                           for lm in face_landmarks.landmark])

            left_eye = pts[LEFT_EYE_IDXS]
            right_eye = pts[RIGHT_EYE_IDXS]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < BLINK_THRESHOLD and prev_eye_open:
                blink_timestamps.append(now)
                prev_eye_open = False
            elif ear > BLINK_THRESHOLD + 0.05:
                prev_eye_open = True

            centroid = pts.mean(axis=0)
            motion_history.append((now, centroid))
            while motion_history and (now - motion_history[0][0] > HISTORY_SECONDS):
                motion_history.popleft()
            while blink_timestamps and (now - blink_timestamps[0] > 2.0):
                blink_timestamps.popleft()

            min_xy = pts.min(axis=0)
            max_xy = pts.max(axis=0)
            x1, y1 = min_xy.astype(int)
            x2, y2 = max_xy.astype(int)
            x1, y1 = max(x1 - 10, 0), max(y1 - 10, 0)
            x2, y2 = min(x2 + 10, w - 1), min(y2 + 10, h - 1)
            face_roi = frame[y1:y2, x1:x2].copy()

        # ==== Liveness update ====
        if now - last_update_time >= UPDATE_INTERVAL:
            motion_conf = 0.0
            pose_conf = 0.0
            blink_conf = 0.0

            if len(motion_history) >= 2:
                centroids = np.array([c for (_, c) in motion_history])
                motion_conf = np.clip(np.linalg.norm(
                    centroids[-1] - centroids[0]) * MOTION_SCALE, 0, 100)
                pose_conf = np.clip(head_pose_variation(
                    centroids) * MOTION_SCALE * 0.5, 0, 100)

            if len(blink_timestamps) > 0:
                blink_conf = min(100, 50 + 25 * len(blink_timestamps))

            liveness_score = 0.5 * motion_conf + 0.3 * blink_conf + 0.2 * pose_conf
            liveness_score = np.clip(liveness_score, 0, 100)

            if face_roi is not None:
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                texture_score = compute_texture_real_score(gray_face)
            else:
                texture_score = 0.0
            realness_score = np.clip(texture_score, 0, 100)

            final_score = np.clip(0.6 * liveness_score +
                                  0.4 * realness_score, 0, 100)
            last_update_time = now

        # ==== Recognition only if liveness ok ====
        recognition_active = False
        if face_found and final_score >= FINAL_SCORE_THRESHOLD:
            recognition_active = True
            faces = app.get(frame)
            for face in faces:
                emb = safe_normalize(face.embedding.astype(np.float32))
                name, score = find_best_match(emb, embeddings, person_ids)
                x1, y1, x2, y2 = map(int, face.bbox)
                color = (0, 255, 0) if score >= SIMILARITY_THRESHOLD else (
                    0, 0, 255)
                if score < SIMILARITY_THRESHOLD:
                    name = "Unknown"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name} ({score*100:.1f}%)"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ==== Display info ====
        cv2.putText(frame, f"Liveness: {liveness_score:5.1f}%", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Final: {final_score:5.1f}%", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Recognition: {'ON' if recognition_active else 'OFF'}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if recognition_active else (0, 0, 255), 2)

        if SHOW_WINDOW:
            cv2.imshow("Face Recognition + Liveness", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()
print("🧠 Recognition stopped.")
