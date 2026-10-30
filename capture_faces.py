import os
import re
import sys

import cv2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")
TOTAL_IMAGES = 30


def next_image_number(student_id):
    pattern = re.compile(rf"User\.{re.escape(student_id)}\.(\d+)\.jpg$", re.IGNORECASE)
    existing_numbers = []
    for filename in os.listdir(DATASET_FOLDER):
        match = pattern.fullmatch(filename)
        if match:
            existing_numbers.append(int(match.group(1)))
    return max(existing_numbers, default=0) + 1


def main():
    student_id = sys.argv[1].strip() if len(sys.argv) > 1 else input("Enter Student ID: ").strip()
    if (
        not student_id.isascii()
        or not student_id.isdecimal()
        or len(student_id) > 10
        or not 0 < int(student_id) <= 2_147_483_647
    ):
        print("ERROR: Student ID must be a positive 32-bit whole number.", file=sys.stderr)
        return 1

    os.makedirs(DATASET_FOLDER, exist_ok=True)
    print(f"Capturing faces for Student ID: {student_id}")

    cascade_path = os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml"
    )
    if not os.path.isfile(cascade_path):
        print(
            "ERROR: OpenCV's face detector file is missing. Reinstall project "
            "dependencies with: python -m pip install -r requirements.txt",
            file=sys.stderr
        )
        return 1

    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        print("ERROR: Could not load the face detector.", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        print("ERROR: Camera could not be opened.", file=sys.stderr)
        return 1

    count = 0
    image_number = next_image_number(student_id)
    try:
        while count < TOTAL_IMAGES:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Camera frame could not be read.", file=sys.stderr)
                return 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(100, 100)
            )

            if len(faces) == 1:
                x, y, width, height = faces[0]
                face = gray[y:y + height, x:x + width]
                face = cv2.resize(face, (200, 200))
                file_path = os.path.join(
                    DATASET_FOLDER,
                    f"User.{student_id}.{image_number}.jpg"
                )
                if not cv2.imwrite(file_path, face):
                    print(f"ERROR: Could not save face image: {file_path}", file=sys.stderr)
                    return 1

                count += 1
                image_number += 1
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"Images: {count}/{TOTAL_IMAGES} (one face at a time; press q to cancel)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"Student: {student_id}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )
            cv2.imshow("Capture Faces", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print(f"ERROR: Capture cancelled after {count}/{TOTAL_IMAGES} images.", file=sys.stderr)
                return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Face capture completed. Student ID: {student_id}; images saved: {count}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
