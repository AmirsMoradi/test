# enroll_person.py
from insightface.app import FaceAnalysis
import numpy as np
import os, pickle, cv2

dataset_dir = "faces"
output_path = "face_database.pkl"

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

db = {}

for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_dir):
        continue

    embs = []
    for file in os.listdir(person_dir):
        path = os.path.join(person_dir, file)
        img = cv2.imread(path)
        if img is None:
            continue
        faces = app.get(img)
        if len(faces) == 0:
            continue
        emb = faces[0].embedding / np.linalg.norm(faces[0].embedding)
        embs.append(emb)

    if len(embs) > 0:
        db[person] = np.mean(embs, axis=0)
        print(f"✅ Added {person} ({len(embs)} images)")

with open(output_path, "wb") as f:
    pickle.dump(db, f)

print(f"\n📦 Database saved as {output_path}")
