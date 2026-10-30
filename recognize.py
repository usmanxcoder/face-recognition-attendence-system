import csv
import os
import sys
from datetime import datetime

import cv2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_FILE = os.path.join(BASE_DIR, "database", "students.csv")
ATTENDANCE_FOLDER = os.path.join(BASE_DIR, "attendance")
ATTENDANCE_FILE = os.path.join(ATTENDANCE_FOLDER, "attendance.csv")
PHOTO_FOLDER = os.path.join(BASE_DIR, "static", "attendance_photos")
MODEL_FILE = os.path.join(BASE_DIR, "trainer", "trainer.yml")
ATTENDANCE_FIELDS = ["Student_ID", "Name", "Date", "Time", "Status", "Photo"]


def load_students():
    students = {}
    if not os.path.exists(STUDENT_FILE):
        return students

    with open(STUDENT_FILE, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            try:
                student_id = int(row.get("Student_ID", ""))
            except (TypeError, ValueError):
                continue
            students[student_id] = row.get("Name", "")
    return students


def mark_attendance(student_id, name, frame):
    today = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, newline="", encoding="utf-8") as file:
            if any(
                row.get("Student_ID") == str(student_id)
                and row.get("Date") == today
                for row in csv.DictReader(file)
            ):
                return False

    photo_name = f"{student_id}_{today}.jpg"
    photo_path = os.path.join(PHOTO_FOLDER, photo_name)
    if not cv2.imwrite(photo_path, frame):
        raise OSError(f"Could not save attendance photo: {photo_path}")

    file_exists = os.path.exists(ATTENDANCE_FILE) and os.path.getsize(ATTENDANCE_FILE) > 0
    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ATTENDANCE_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "Student_ID": student_id,
            "Name": name,
            "Date": today,
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Status": "Present",
            "Photo": f"static/attendance_photos/{photo_name}"
        })

    print(f"Attendance marked: {name} ({student_id})")
    return True


def main():
    os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)
    os.makedirs(PHOTO_FOLDER, exist_ok=True)

    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        print(
            "ERROR: OpenCV face recognizer is unavailable. Install opencv-contrib-python.",
            file=sys.stderr
        )
        return 1
    if not os.path.isfile(MODEL_FILE):
        print("ERROR: Trained model not found. Train the model first.", file=sys.stderr)
        return 1

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

    students = load_students()
    if not students:
        print("ERROR: No valid students found. Add a student and capture face images first.", file=sys.stderr)
        return 1

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read(MODEL_FILE)
    except cv2.error as error:
        print(f"ERROR: Could not load the trained model: {error}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        print("ERROR: Camera could not be opened.", file=sys.stderr)
        return 1

    print("Face recognition started. Press q in the camera window to stop.")
    try:
        while True:
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

            for x, y, width, height in faces:
                face = gray[y:y + height, x:x + width]
                face = cv2.resize(face, (200, 200))
                student_id, confidence = recognizer.predict(face)
                recognized = confidence < 70 and student_id in students

                if recognized:
                    name = students[student_id]
                    color = (0, 255, 0)
                    mark_attendance(student_id, name, frame)
                    label = f"{name} (ID: {student_id})"
                else:
                    color = (0, 0, 255)
                    label = "Unknown"

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    color,
                    2
                )
                cv2.putText(
                    frame,
                    label,
                    (x, max(y - 15, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

            cv2.imshow("Face Recognition Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except (cv2.error, OSError) as error:
        print(f"ERROR: Face recognition stopped: {error}", file=sys.stderr)
        return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Face recognition stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
