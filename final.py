import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import pickle
from insightface.app import FaceAnalysis

# ==== Base settings ====
RTSP_URL = "rtsp://admin:Qwerty123@10.41.41.11:554/cam/realmonitor?channel=1&subtype=0"

RESIZE_WIDTH = 640
FRAME_SKIP = 1
SHOW_WINDOW = True

UPDATE_INTERVAL = 0.5
HISTORY_SECONDS = 1.2
MOTION_SCALE = 8000.0
FINAL_THRESHOLD = 80.0  # برای فعال شدن تشخیص هویت

# ==== Mediapipe ====
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# ==== Face recognition model ====
print("⏳ Loading InsightFace (GPU if possible)...")
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("✅ InsightFace initialized.")

# ==== Load face database ====
db_names = []
db_embs = None
db_loaded = False

try:
    with open("face_database.pkl", "rb") as f:
        face_db = pickle.load(f)
    db_names = list(face_db.keys())
    db_embs = np.array([face_db[name] for name in db_names], dtype=np.float32)
    db_embs /= (np.linalg.norm(db_embs, axis=1, keepdims=True) + 1e-6)
    db_loaded = True
    print(f"📂 Loaded {len(db_names)} persons from face_database.pkl")
except FileNotFoundError:
    print("⚠️ face_database.pkl not found. Recognition disabled.")
    db_loaded = False


def find_best_match(emb):
    if not db_loaded or db_embs is None or len(db_embs) == 0:
        return "Unknown", 0.0
    sims = db_embs @ emb
    idx = int(np.argmax(sims))
    return db_names[idx], float(sims[idx])


# ==== Helper functions ====
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


# ==== Video Capture ====
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
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

last_name = "Unknown"
last_sim = 0.0
last_recog_time = 0.0
RECOG_INTERVAL = 1.0

print("✅ GPU Liveness + Recognition pipeline started...")

with mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
) as face_mesh:

    prev_eye_open = True
    face_found = False
    face_roi = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame not received from RTSP")
            time.sleep(0.3)
            continue

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
            lm = results.multi_face_landmarks[0]
            pts = np.array([(p.x * w, p.y * h) for p in lm.landmark])

            left_eye = pts[LEFT_EYE_IDXS]
            right_eye = pts[RIGHT_EYE_IDXS]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

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
            x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
            x2, y2 = min(w - 1, x2 + 10), min(h - 1, y2 + 10)
            if x2 > x1 and y2 > y1:
                face_roi = frame[y1:y2, x1:x2].copy()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

        else:
            face_found = False
            face_roi = None
            motion_history.clear()
            blink_timestamps.clear()
            liveness_score = 0.0
            realness_score = 0.0
            final_score = 0.0
            last_name = "Unknown"
            last_sim = 0.0

        # --- Update every interval ---
        if now - last_update_time >= UPDATE_INTERVAL:
            if face_found:
                motion_conf = 0.0
                pose_conf = 0.0
                blink_conf = 0.0

                if len(motion_history) >= 2:
                    centroids = np.array([c for (_, c) in motion_history])
                    motion_conf = np.clip(
                        np.linalg.norm(centroids[-1] - centroids[0]) * MOTION_SCALE, 0, 100)
                    pose_conf = np.clip(
                        head_pose_variation(centroids) * MOTION_SCALE * 0.5, 0, 100)
                if len(blink_timestamps) > 0:
                    blink_conf = min(100, 50 + 25 * len(blink_timestamps))

                liveness_score = np.clip(
                    0.5 * motion_conf + 0.3 * blink_conf + 0.2 * pose_conf, 0, 100)

                if face_roi is not None:
                    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    texture_score = compute_texture_real_score(gray)
                    realness_score = texture_score
                else:
                    realness_score = 0.0

                final_score = np.clip(0.6 * liveness_score + 0.4 * realness_score, 0, 100)
            else:
                liveness_score = 0.0
                realness_score = 0.0
                final_score = 0.0

            last_update_time = now

        # --- Recognition only if real and interval passed ---
        if (
            db_loaded
            and face_found
            and final_score >= FINAL_THRESHOLD
            and (now - last_recog_time) >= RECOG_INTERVAL
        ):
            faces = face_app.get(frame)
            if faces:
                face = faces[0]
                emb = face.embedding.astype(np.float32)
                emb /= (np.linalg.norm(emb) + 1e-6)
                name, sim = find_best_match(emb)
                if sim < 0.45:
                    name = "Unknown"
                last_name = name
                last_sim = sim
                last_recog_time = now

        # --- Draw overlay ---
        cv2.putText(frame, f"Liveness: {liveness_score:5.1f}%",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Realness: {realness_score:5.1f}%",
                    (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)
        cv2.putText(frame, f"FINAL: {final_score:5.1f}%",
                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"Name: {last_name} ({last_sim*100:.1f}%)",
                    (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if SHOW_WINDOW:
            cv2.imshow("Liveness + Name (GPU)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()
print("🧠 Liveness + Name pipeline stopped.")
