import cv2
import numpy as np
import sqlite3
import insightface
import time

# ===============================
# تنظیمات پایه
# ===============================
RTSP_URL = "rtsp://admin:Qwerty123@10.41.41.11:554/cam/realmonitor?channel=1&subtype=0"
DB_PATH = "db/faces.db"
SIMILARITY_THRESHOLD = 0.45   # حداقل شباهت برای شناسایی
RESIZE_WIDTH = 480            # کاهش اندازه تصویر برای سرعت بیشتر
FRAME_SKIP = 2                # هر دو فریم یکبار تحلیل
SHOW_WINDOW = True            # نمایش خروجی در پنجره

# ===============================
# بارگذاری مدل InsightFace (فقط بخش‌های لازم)
# ===============================
print("⏳ Loading optimized InsightFace (buffalo_l) ...")
app = insightface.app.FaceAnalysis(
    name="buffalo_l",
    allowed_modules=['detection', 'recognition']  # فقط تشخیص و بازشناسی
)
app.prepare(ctx_id=-1)  # استفاده از CoreML در macOS یا CPU در بقیه
print("✅ Model loaded (CoreML/CPU auto).\n")

# ===============================
# بارگذاری Embeddings از دیتابیس
# ===============================
print("📂 Loading embeddings from database...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

persons = {row[0]: row[1]
           for row in cursor.execute("SELECT id, name FROM persons")}
embeddings = []
person_ids = []


def safe_normalize(v):
    """نرمال‌سازی L2 به‌صورت ایمن"""
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
print(
    f"✅ Loaded {len(embeddings)} embeddings for {len(set(person_ids))} persons.\n")
conn.close()

# ===============================
# تابع تطبیق چهره جدید با دیتابیس
# ===============================


def find_best_match(new_emb, embeddings, person_ids):
    if len(embeddings) == 0:
        return None, 0.0
    new_emb = safe_normalize(new_emb)
    sims = np.dot(embeddings, new_emb)  # شباهت کسینوسی
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    best_person = persons.get(person_ids[best_idx], "Unknown")
    return best_person, best_score


# ===============================
# شروع استریم RTSP
# ===============================
print("🎥 Connecting to RTSP stream...")
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open RTSP stream")

print("✅ Stream opened. Starting recognition...\n")

frame_count = 0
fps_start = time.time()
fps_counter = 0

# ===============================
# حلقه اصلی پردازش
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue  # پرش از فریم برای کاهش بار پردازش

    # تغییر اندازه فریم برای افزایش سرعت
    h, w = frame.shape[:2]
    scale = RESIZE_WIDTH / float(w)
    frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)

    for face in faces:
        emb = safe_normalize(face.embedding.astype(np.float32))
        name, score = find_best_match(emb, embeddings, person_ids)

        x1, y1, x2, y2 = map(int, face.bbox)
        color = (0, 255, 0) if score >= SIMILARITY_THRESHOLD else (0, 0, 255)
        if score < SIMILARITY_THRESHOLD:
            name = "Unknown"

        label = f"{name} ({score*100:.1f}%)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # محاسبه FPS هر ثانیه
    fps_counter += 1
    if time.time() - fps_start >= 1.0:
        fps = fps_counter / (time.time() - fps_start)
        fps_start = time.time()
        fps_counter = 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if SHOW_WINDOW:
        cv2.imshow("Face Recognition (RTSP, Optimized)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
print("🧠 Optimized recognition stopped.")
