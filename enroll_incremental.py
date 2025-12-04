import os
import cv2
import numpy as np
import sqlite3
import insightface
from datetime import datetime

# ===============================
# تنظیمات کلی
# ===============================
FACES_DIR = "faces"
DB_PATH = "db/faces.db"

# ===============================
# بارگذاری مدل InsightFace
# ===============================
print("⏳ Loading InsightFace (buffalo_l) ...")
app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)   # GPU=0 ، اگر CPU داری: ctx_id=-1
print("✅ Model loaded successfully.\n")

# ===============================
# آماده‌سازی دیتابیس SQLite
# ===============================
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH)
conn.execute("""
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    created_at TEXT
)
""")
conn.execute("""
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    embedding BLOB,
    image_path TEXT UNIQUE,
    created_at TEXT
)
""")
conn.commit()

# ===============================
# تابع کمکی
# ===============================


def l2_normalize(v):
    return v / np.linalg.norm(v)


def get_all_image_paths():
    """بررسی کل تصاویر موجود در faces"""
    all_paths = []
    for person_name in os.listdir(FACES_DIR):
        person_dir = os.path.join(FACES_DIR, person_name)
        if os.path.isdir(person_dir):
            for img_file in os.listdir(person_dir):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    all_paths.append(
                        (person_name, os.path.join(person_dir, img_file)))
    return all_paths

# ===============================
# تابع اصلی پردازش
# ===============================


def process_new_images():
    new_count = 0
    all_images = get_all_image_paths()

    # مسیرهای موجود در دیتابیس
    existing = set(
        row[0] for row in conn.execute("SELECT image_path FROM embeddings").fetchall()
    )

    for person_name, img_path in all_images:
        if img_path in existing:
            continue  # این تصویر قبلاً پردازش شده

        # بررسی وجود فرد در جدول persons
        conn.execute("INSERT OR IGNORE INTO persons (name, created_at) VALUES (?, ?)",
                     (person_name, datetime.now().isoformat()))
        person_id = conn.execute(
            "SELECT id FROM persons WHERE name=?", (person_name,)).fetchone()[0]

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Cannot read image: {img_path}")
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print(f"⚠️ No face detected in: {img_path}")
            continue

        face = faces[0]
        emb = l2_normalize(face.embedding.astype(np.float32))
        emb_bytes = emb.tobytes()

        conn.execute("""
            INSERT OR IGNORE INTO embeddings (person_id, embedding, image_path, created_at)
            VALUES (?, ?, ?, ?)
        """, (person_id, emb_bytes, img_path, datetime.now().isoformat()))
        conn.commit()

        print(
            f"✅ Added new embedding: {person_name} → {os.path.basename(img_path)}")
        new_count += 1

    if new_count == 0:
        print("📁 No new images found.")
    else:
        print(f"\n🎉 Added {new_count} new image(s) to database.")


# ===============================
# اجرای اصلی
# ===============================
if not os.path.exists(FACES_DIR):
    print(f"❌ Folder not found: {FACES_DIR}")
else:
    process_new_images()

conn.close()
print(f"\n✅ Database updated successfully: {DB_PATH}")
