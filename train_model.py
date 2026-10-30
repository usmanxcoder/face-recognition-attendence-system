import csv
import os
import re
import sys

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")
STUDENT_FILE = os.path.join(BASE_DIR, "database", "students.csv")
TRAINER_FOLDER = os.path.join(BASE_DIR, "trainer")
MODEL_FILE = os.path.join(TRAINER_FOLDER, "trainer.yml")
IMAGE_PATTERN = re.compile(r"User\.(\d+)\.\d+\.jpg$", re.IGNORECASE)


def load_registered_student_ids():
    if not os.path.isfile(STUDENT_FILE):
        return set()

    student_ids = set()
    with open(STUDENT_FILE, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            try:
                student_id = int(row.get("Student_ID", ""))
            except (TypeError, ValueError):
                continue
            if 0 < student_id <= 2_147_483_647:
                student_ids.add(student_id)
    return student_ids


def load_images_and_labels(registered_student_ids):
    faces = []
    labels = []
    unregistered_ids = set()

    for filename in sorted(os.listdir(DATASET_FOLDER)):
        match = IMAGE_PATTERN.fullmatch(filename)
        if not match:
            continue

        student_id = int(match.group(1))
        if student_id > 2_147_483_647:
            print(f"WARNING: Skipped image with out-of-range student ID: {filename}", file=sys.stderr)
            continue
        if student_id not in registered_student_ids:
            unregistered_ids.add(student_id)
            continue

        image_path = os.path.join(DATASET_FOLDER, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            print(f"WARNING: Skipped unreadable image: {filename}", file=sys.stderr)
            continue

        faces.append(cv2.resize(image, (200, 200)))
        labels.append(student_id)

    for student_id in sorted(unregistered_ids):
        print(
            f"WARNING: Ignored dataset images for unregistered student ID {student_id}.",
            file=sys.stderr
        )

    return faces, labels


def main():
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    os.makedirs(TRAINER_FOLDER, exist_ok=True)

    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        print(
            "ERROR: OpenCV face recognizer is unavailable. Install opencv-contrib-python.",
            file=sys.stderr
        )
        return 1

    registered_student_ids = load_registered_student_ids()
    if not registered_student_ids:
        print("ERROR: No valid students are registered.", file=sys.stderr)
        return 1

    faces, labels = load_images_and_labels(registered_student_ids)
    if not faces:
        print(
            "ERROR: No face images for registered students were found. Capture faces first.",
            file=sys.stderr
        )
        return 1

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.train(faces, np.asarray(labels, dtype=np.int32))
        temporary_model = os.path.join(TRAINER_FOLDER, "trainer.tmp.yml")
        recognizer.save(temporary_model)
        os.replace(temporary_model, MODEL_FILE)
    except (cv2.error, OSError, ValueError) as error:
        print(f"ERROR: Model training failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Training completed. Images: {len(faces)}; students: {len(set(labels))}; "
        f"model: {MODEL_FILE}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
