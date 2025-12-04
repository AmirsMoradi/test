import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# ==== Base settings ====
RTSP_URL = "rtsp://admin:Qwerty123@10.41.41.11:554/cam/realmonitor?channel=1&subtype=0"

RESIZE_WIDTH = 640
FRAME_SKIP = 1
SHOW_WINDOW = True

UPDATE_INTERVAL = 0.5
HISTORY_SECONDS = 1.2
MOTION_SCALE = 8000.0

# ==== Anti-spoof model (optional CNN) ====
USE_ANTISPOOF_MODEL = False  # set True when you have an ONNX model
ANTISPOOF_MODEL_PATH = "antispoof.onnx"

anti_spoof_net = None
if USE_ANTISPOOF_MODEL:
    anti_spoof_net = cv2.dnn.readNetFromONNX(ANTISPOOF_MODEL_PATH)

# ==== Mediapipe ====
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# ==== Helper functions ====


def eye_aspect_ratio(eye_pts):
    """Compute Eye Aspect Ratio for blink detection."""
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def head_pose_variation(centroids):
    """Rough head motion estimate based on centroid changes."""
    if len(centroids) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(centroids)):
        total += np.linalg.norm(centroids[i] - centroids[i - 1])
    return total / (len(centroids) - 1)


def compute_texture_real_score(face_gray):
    """
    Heuristic texture-based realness score (0-100).

    Idea:
    - Compute Laplacian variance (focus / high-frequency energy).
    - Map it into [0, 100] using a rough empirical scale.
    WARNING: Heuristic, you should tune thresholds for your camera.
    """
    face_gray = cv2.resize(face_gray, (128, 128))
    lap = cv2.Laplacian(face_gray, cv2.CV_64F)
    var = lap.var()

    # Heuristic mapping: adjust these numbers per camera/scene
    # typical real faces (in focus) might be around 50-200 on this scale
    norm = (var - 40.0) / 120.0  # shift + scale
    norm = np.clip(norm, 0.0, 1.0)
    return float(norm * 100.0)


def run_antispoof_cnn(face_bgr):
    """
    Optional: run anti-spoof CNN (ONNX) if available.
    You must adapt preprocessing according to your model.
    Here we assume:
      - input: 224x224 RGB, normalized to [0,1]
      - output: [prob_spoof, prob_real] or [real_prob]
    """
    if anti_spoof_net is None:
        return None

    img = cv2.resize(face_bgr, (224, 224))
    blob = img.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC -> CHW
    blob = np.expand_dims(blob, axis=0)

    anti_spoof_net.setInput(blob)
    out = anti_spoof_net.forward()

    # You must adapt this depending on model output shape
    # Example 1: out shape (1, 2): [spoof_prob, real_prob]
    if out.shape[-1] == 2:
        spoof_prob, real_prob = out[0]
    else:
        # Example 2: out shape (1, 1): real_prob
        real_prob = out[0][0]
        spoof_prob = 1.0 - real_prob

    real_prob = float(np.clip(real_prob, 0.0, 1.0))
    return real_prob * 100.0


# ==== Video Capture ====
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("❌ Cannot open RTSP stream")

frame_count = 0
motion_history = deque()
blink_timestamps = deque()
last_update_time = time.time()
liveness_score = 0.0
final_score = 0.0
realness_score = 0.0

# FaceMesh indices
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
MOUTH_IDXS = [78, 81, 13, 311, 308, 402]

BLINK_THRESHOLD = 0.2
BLINK_RECOVERY_TIME = 0.25

print("✅ Liveness + Anti-spoof pipeline started...")

with mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
) as face_mesh:

    prev_eye_open = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame not received from RTSP")
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

            # eyes & mouth
            left_eye = pts[LEFT_EYE_IDXS]
            right_eye = pts[RIGHT_EYE_IDXS]
            mouth = pts[MOUTH_IDXS]

            # eye aspect ratio
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # blink detection
            if ear < BLINK_THRESHOLD and prev_eye_open:
                blink_timestamps.append(now)
                prev_eye_open = False
            elif ear > BLINK_THRESHOLD + 0.05:
                prev_eye_open = True

            # centroid
            centroid = pts.mean(axis=0)
            motion_history.append((now, centroid))
            while motion_history and (now - motion_history[0][0] > HISTORY_SECONDS):
                motion_history.popleft()

            while blink_timestamps and (now - blink_timestamps[0] > 2.0):
                blink_timestamps.popleft()

            # rough face bbox for ROI
            min_xy = pts.min(axis=0)
            max_xy = pts.max(axis=0)
            x1, y1 = min_xy.astype(int)
            x2, y2 = max_xy.astype(int)
            x1 = max(x1 - 10, 0)
            y1 = max(y1 - 10, 0)
            x2 = min(x2 + 10, w - 1)
            y2 = min(y2 + 10, h - 1)

            if x2 > x1 and y2 > y1:
                face_roi = frame[y1:y2, x1:x2].copy()

            # draw landmarks
            if SHOW_WINDOW:
                mp_draw.draw_landmarks(
                    frame, face_landmarks, mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_draw.DrawingSpec(
                        color=(0, 255, 0), thickness=1, circle_radius=1
                    )
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

        else:
            motion_history.clear()
            blink_timestamps.clear()

        # ---- update every UPDATE_INTERVAL seconds ----
        if now - last_update_time >= UPDATE_INTERVAL:
            motion_conf = 0.0
            pose_conf = 0.0
            blink_conf = 0.0

            if len(motion_history) >= 2:
                centroids = np.array([c for (_, c) in motion_history])
                motion_conf = np.clip(
                    np.linalg.norm(centroids[-1] -
                                   centroids[0]) * MOTION_SCALE,
                    0, 100
                )
                pose_conf = np.clip(
                    head_pose_variation(centroids) * MOTION_SCALE * 0.5,
                    0, 100
                )

            if len(blink_timestamps) > 0:
                blink_conf = min(100, 50 + 25 * len(blink_timestamps))

            # liveness from motion + blink + pose
            liveness_score = 0.5 * motion_conf + 0.3 * blink_conf + 0.2 * pose_conf
            liveness_score = np.clip(liveness_score, 0, 100)

            # --- Anti-spoof / realness ---
            texture_score = 0.0
            cnn_real_score = None

            if face_roi is not None:
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                texture_score = compute_texture_real_score(gray_face)
                if USE_ANTISPOOF_MODEL:
                    cnn_real_score = run_antispoof_cnn(face_roi)

            # fuse texture + cnn (if available)
            if cnn_real_score is not None:
                realness_score = 0.6 * cnn_real_score + 0.4 * texture_score
            else:
                realness_score = texture_score  # only heuristic for now

            realness_score = np.clip(realness_score, 0, 100)

            # final combined score: how much we trust this is a real, live face
            final_score = 0.6 * liveness_score + 0.4 * realness_score
            final_score = np.clip(final_score, 0, 100)

            last_update_time = now

        # ---- draw overlay ----
        text1 = f"Liveness: {liveness_score:5.1f}%"
        text2 = f"Realness: {realness_score:5.1f}%"
        text3 = f"FINAL: {final_score:5.1f}%"

        cv2.putText(frame, text1, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, text2, (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)
        cv2.putText(frame, text3, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if SHOW_WINDOW:
            cv2.imshow("Liveness + Anti-Spoof", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()
