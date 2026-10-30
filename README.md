# Face Recognition Attendance System

A local Flask application for managing students and recording attendance with
OpenCV face recognition. It provides a web dashboard for student and attendance
records, while face capture and recognition use a webcam connected to the
machine running the application.

## Features

- Add, edit, search, and remove student records.
- Capture face samples for a student using a webcam.
- Train an OpenCV LBPH face-recognition model.
- Record attendance once per student per day and save an attendance photo.
- Review attendance records and download them as a CSV file.
- View basic application and data status.

## Requirements

- Python 3.10 or later
- A webcam and a graphical desktop session for face capture and recognition

## Setup

From the project directory, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the application:

```powershell
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. The
application creates its local data folders and empty CSV files as needed.

## Typical workflow

1. Add a student from the dashboard.
2. Capture face samples for that student. Keep the webcam connected; the app
   captures samples and trains the model.
3. Start face recognition attendance from the dashboard. Press `q` in the
   OpenCV camera window to stop.
4. Review the attendance page or download the CSV report.

## Local data and privacy

Student details, attendance records, captured face images, attendance photos,
and the trained recognition model are local application data. They are
intentionally excluded from Git by `.gitignore`; do not commit or publish them.
The application recreates the required folders and CSV files when it starts.

Face images and attendance data are sensitive personal information. Obtain
appropriate consent, restrict access to the machine and records, provide a
manual way to verify attendance, and delete data when it is no longer needed.

This project is intended for local development. Do not expose the Flask
development server or its webcam/data endpoints to an untrusted network.
