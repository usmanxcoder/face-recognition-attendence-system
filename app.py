from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory
)

import csv
import glob
import os
import subprocess
import sys
from datetime import datetime

app = Flask(__name__)
app.secret_key = "face_attendance_secret"

# ==============================
# Folders
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATTENDANCE_FOLDER = os.path.join(BASE_DIR, "attendance")
DATABASE_FOLDER = os.path.join(BASE_DIR, "database")
PHOTO_FOLDER = os.path.join(BASE_DIR, "static", "attendance_photos")
DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")
TRAINER_FOLDER = os.path.join(BASE_DIR, "trainer")
MODEL_FILE = os.path.join(TRAINER_FOLDER, "trainer.yml")

ATTENDANCE_FILE = os.path.join(
    ATTENDANCE_FOLDER,
    "attendance.csv"
)

STUDENT_FILE = os.path.join(
    DATABASE_FOLDER,
    "students.csv"
)

# ==============================
# CSV Helpers
# ==============================


def read_csv_rows(path):
    if not os.path.exists(path):
        return []

    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: row.get(key, "")
                for key in fieldnames
            })


def run_python_script(script_name, *arguments):
    script_path = os.path.join(BASE_DIR, script_name)
    try:
        result = subprocess.run(
            [sys.executable, script_path, *arguments],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=False
        )
    except OSError as error:
        return False, f"Could not start {script_name}: {error}"

    output = (result.stderr or result.stdout or "").strip()
    if result.returncode != 0:
        return False, output or f"{script_name} failed."
    return True, output


def is_valid_student_id(student_id):
    if not student_id.isascii() or not student_id.isdecimal() or len(student_id) > 10:
        return False
    return 0 < int(student_id) <= 2_147_483_647


def prepare_attendance_rows(rows):
    for row in rows:
        photo = str(row.get("Photo", "") or "").strip()
        row["Photo_URL"] = (
            url_for("photos", filename=os.path.basename(photo))
            if photo
            else ""
        )
    return rows


# ==============================
# Create Required Folders
# ==============================

os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)
os.makedirs(PHOTO_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# ==============================
# Create students.csv
# ==============================

if not os.path.exists(STUDENT_FILE):
    write_csv_rows(STUDENT_FILE, ["Student_ID", "Name"], [])

# ==============================
# Create attendance.csv
# ==============================

if not os.path.exists(ATTENDANCE_FILE):
    write_csv_rows(
        ATTENDANCE_FILE,
        ["Student_ID", "Name", "Date", "Time", "Status", "Photo"],
        []
    )

# ==============================
# Dashboard
# ==============================

@app.route("/")
def home():

    students = read_csv_rows(STUDENT_FILE)
    attendance = read_csv_rows(ATTENDANCE_FILE)

    total_students = len(students)

    today = datetime.now().strftime("%Y-%m-%d")
    today_attendance = sum(
        1 for record in attendance if str(record.get("Date", "")) == today
    )

    return render_template(
        "index.html",
        attendance=prepare_attendance_rows(attendance),
        total_students=total_students,
        today_attendance=today_attendance,
        today=today
    )
# ==========================================
# Start Face Recognition Attendance
# ==========================================

@app.route("/start", methods=["POST"])
def start_attendance():
    if not os.path.isfile(MODEL_FILE):
        flash("Train the recognition model before starting attendance.", "danger")
        return redirect(url_for("home"))

    try:
        subprocess.Popen(
            [sys.executable, os.path.join(BASE_DIR, "recognize.py")],
            cwd=BASE_DIR
        )
    except OSError as error:
        flash(f"Could not start face recognition: {error}", "danger")
    else:
        flash("Face recognition was launched. Close its camera window to stop.", "success")

    return redirect(url_for("home"))


# ==========================================
# Capture Faces
# ==========================================

@app.route("/capture", methods=["POST"])
def capture_faces():
    flash("Choose a student and use their Capture action.", "info")
    return redirect(url_for("students"))


# ==========================================
# Train Model
# ==========================================

@app.route("/train", methods=["POST"])
def train_model():
    success, output = run_python_script("train_model.py")
    if success:
        flash(output or "Face recognition model trained successfully.", "success")
    else:
        flash(f"Model training failed: {output}", "danger")

    return redirect(url_for("students"))


# ==========================================
# Students List
# ==========================================

@app.route("/students")
def students():

    rows = read_csv_rows(STUDENT_FILE)
    search = (request.args.get("search", "") or "").strip().lower()

    if search:
        rows = [
            row for row in rows
            if search in str(row.get("Student_ID", "")).lower()
            or search in str(row.get("Name", "")).lower()
        ]

    return render_template(
        "students.html",
        table=rows,
        total_students=len(rows)
    )
# ==========================================
# Add Student
# ==========================================

@app.route("/add_student", methods=["GET", "POST"])
def add_student():

    if request.method == "POST":
        student_id = request.form.get("id", "").strip()
        student_name = request.form.get("name", "").strip()

        if not is_valid_student_id(student_id):
            flash("Student ID must be a positive whole number.", "danger")
            return redirect(url_for("add_student"))
        student_id = str(int(student_id))
        if not student_name:
            flash("Student name is required.", "danger")
            return redirect(url_for("add_student"))

        rows = read_csv_rows(STUDENT_FILE)

        if any(str(row.get("Student_ID", "")).strip() == student_id for row in rows):
            flash("Student ID already exists.", "danger")
            return redirect(url_for("add_student"))

        rows.append({"Student_ID": student_id, "Name": student_name})
        write_csv_rows(STUDENT_FILE, ["Student_ID", "Name"], rows)

        flash(
            "Student added. Use Capture beside the student to collect face images and train the model.",
            "success"
        )

        return redirect(url_for("students"))

    return render_template("add_student.html")

# ==========================================
# Edit Student
# ==========================================

@app.route("/edit_student/<student_id>", methods=["GET", "POST"])
def edit_student(student_id):

    rows = read_csv_rows(STUDENT_FILE)
    student = next(
        (row for row in rows if str(row.get("Student_ID", "")).strip() == str(student_id)),
        None
    )
    if student is None:
        flash("Student not found.", "danger")
        return redirect(url_for("students"))

    if request.method == "POST":

        new_name = request.form.get("name", "").strip()
        if not new_name:
            flash("Student name is required.", "danger")
            return redirect(url_for("edit_student", student_id=student_id))

        student["Name"] = new_name

        write_csv_rows(STUDENT_FILE, ["Student_ID", "Name"], rows)

        flash("Student updated successfully.", "success")
        return redirect(url_for("students"))

    return render_template("edit_student.html", student=student)
@app.route("/capture/<student_id>", methods=["POST"])
def capture_student(student_id):
    if not is_valid_student_id(student_id):
        flash("Invalid student ID.", "danger")
        return redirect(url_for("students"))

    rows = read_csv_rows(STUDENT_FILE)
    if not any(str(row.get("Student_ID", "")).strip() == student_id for row in rows):
        flash("Student not found.", "danger")
        return redirect(url_for("students"))

    captured, capture_output = run_python_script("capture_faces.py", student_id)
    if not captured:
        flash(f"Face capture failed: {capture_output}", "danger")
        return redirect(url_for("students"))

    trained, training_output = run_python_script("train_model.py")
    if not trained:
        flash(f"Face images were captured, but model training failed: {training_output}", "danger")
        return redirect(url_for("students"))

    flash("Face images captured and model trained successfully.", "success")

    return redirect(url_for("students"))

# ==========================================
# Delete Student
# ==========================================

@app.route("/delete_student/<student_id>", methods=["POST"])
def delete_student(student_id):
    if not is_valid_student_id(student_id):
        flash("Invalid student ID.", "danger")
        return redirect(url_for("students"))

    rows = read_csv_rows(STUDENT_FILE)
    if not any(str(row.get("Student_ID", "")).strip() == student_id for row in rows):
        flash("Student not found.", "danger")
        return redirect(url_for("students"))

    rows = [
        row for row in rows
        if str(row.get("Student_ID", "")).strip() != str(student_id)
    ]

    write_csv_rows(STUDENT_FILE, ["Student_ID", "Name"], rows)

    for image_path in glob.glob(os.path.join(DATASET_FOLDER, f"User.{student_id}.*.jpg")):
        os.remove(image_path)

    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)

    if any(name.lower().endswith(".jpg") for name in os.listdir(DATASET_FOLDER)):
        trained, output = run_python_script("train_model.py")
        if not trained:
            flash(f"Student deleted, but model retraining failed: {output}", "danger")
            return redirect(url_for("students"))

    flash("Student deleted and recognition data updated.", "success")

    return redirect(url_for("students"))
# ==========================================
# Attendance Report
# ==========================================

@app.route("/attendance")
def attendance():

    rows = prepare_attendance_rows(read_csv_rows(ATTENDANCE_FILE))

    return render_template(
        "attendance.html",
        attendance=rows
    )


# ==========================================
# Download Attendance CSV
# ==========================================

@app.route("/download")
def download():

    return send_from_directory(

        ATTENDANCE_FOLDER,

        "attendance.csv",

        as_attachment=True

    )


# ==========================================
# Attendance Photos
# ==========================================

@app.route("/photos/<filename>")
def photos(filename):

    return send_from_directory(

        PHOTO_FOLDER,

        filename

    )
# ==========================================
# 404 Error Page
# ==========================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(

        "404.html"

    ), 404


# ==========================================
# 500 Error Page
# ==========================================

@app.errorhandler(500)
def internal_error(error):

    flash(

        "Internal Server Error.",

        "danger"

    )

    return redirect(

        url_for("home")

    )


# ==========================================
# Refresh Attendance
# ==========================================

@app.route("/refresh")
def refresh():

    return redirect(

        url_for("home")

    )


# ==========================================
# About Project
# ==========================================

@app.route("/about")
def about():

    return render_template(

        "about.html"

    )


# ==========================================
# System Information
# ==========================================

@app.route("/system")
def system():

    students = read_csv_rows(STUDENT_FILE)
    attendance = read_csv_rows(ATTENDANCE_FILE)

    info = {
        "Total Students": len(students),
        "Attendance Records": len(attendance),
        "Dataset Images": len(os.listdir(DATASET_FOLDER)) if os.path.exists(DATASET_FOLDER) else 0,
        "Model Exists": os.path.exists(MODEL_FILE)
    }

    return render_template(
        "system.html",
        info=info
    )


# ==========================================
# Run Flask App
# ==========================================

if __name__ == "__main__":

    app.run(

        debug=True,

        host="0.0.0.0",

        port=5000

    )
    