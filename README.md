# Smart Attendance System using Face Recognition

> A production-ready Computer Vision project that marks attendance in real-time from a webcam feed, prevents duplicate entries, and logs everything to timestamped CSV files.

---

## Features

| Feature | Details |
|---|---|
| **Face Detection** | OpenCV + HOG model (via `face_recognition`) |
| **Face Recognition** | 128-d encoding comparison (dlib deep-metric learning) |
| **Student Registration** | New streamlined flow for name, reg. number, and photo |
| **Attendance logging** | Daily CSV files with **Name, RegNo, Date, Time, Status** |
| **Duplicate detection** | Same person cannot be marked twice per calendar day |
| **Cooldown window** | 5-second gap between recognition events (prevents spam) |
| **Real-time webcam** | Live bounding boxes with name and RegNo labels |
| **CLI modes** | `--register`, `--train`, `--summary`, `--logs` flags |

---

## Folder Structure

```
smart-attendance-cv/
│
├── dataset/                   ← Training images (folder: REGNO_Name)
│   ├── CS2023001_Alice/
│   │   └── img_001.jpg
│   └── CS2023002_Bob/
│       └── img_001.jpg
│
├── src/
│   ├── __init__.py
│   ├── register_student.py    ← Handles student registration & validation
│   ├── train.py               ← Encodes dataset images → face_encodings.pkl
│   ├── recognize.py           ← Real-time webcam recognition loop
│   └── attendance.py          ← CSV logging + duplicate detection
│
├── attendance_logs/           ← Auto-created; one CSV per day
│   └── attendance_YYYY-MM-DD.csv
│
├── main.py                    ← Entry point
├── requirements.txt
└── README.md
```

---

## Prerequisites

- **Python:** 3.9 – 3.12
- **Webcam:** Any USB or built-in camera
- **OS:** Windows 10/11 (Preferred), Ubuntu, or macOS

---

## 🛠 Detailed Setup (Windows Only)

`dlib` (the engine behind face recognition) is difficult to install on Windows. Follow **ONE** of these two options:

### Option A: Using Visual Studio (Recommended if you have it)
1.  Install **Visual Studio Community** with the **"Desktop development with C++"** workload.
2.  Open the **"Developer Command Prompt for VS 2022"** from your Start Menu.
3.  Navigate to the project folder and run:
    ```cmd
    pip install cmake
    pip install dlib
    ```

### Option B: Using Pre-built Wheels (Faster, no VS required)
1.  Check your Python version: `python --version`
2.  Download the `.whl` matching your version from [this link](https://github.com/z-mahmud22/Dlib_Windows_Python3.x/releases).
3.  Install it directly:
    ```cmd
    # Example for Python 3.11
    pip install dlib-19.24.4-cp311-cp311-win_amd64.whl
    ```

---

## 🚀 Execution Guide

### 1. Install remaining dependencies
Once `dlib` is installed via Option A or B above, run:
```bash
pip install -r requirements.txt
```

### 2. Register Students (Add Photos & Info)
Instead of manually creating folders, use the built-in registration command:
```bash
python main.py --register
```
It will ask for:
- **Name:** (e.g., Alice Smith)
- **Registration Number:** (e.g., CS2023001)
- **Photo Path:** (Path to a clear picture of the student)

**Note:** The system validates the photo to ensure a face is actually detectable before saving.

### 3. Train the System
If you didn't auto-train during registration, run:
```bash
python main.py --train
```

### 4. Run Attendance System
```bash
python main.py
```
- Stand in front of the webcam.
- Once recognized, your name and RegNo will appear in green.
- **Keys:** `Q` to Quit, `S` to show a summary in the console.

---

## How to Run (CLI Quick Reference)

| Command | Action |
|---|---|
| `python main.py --register` | Interactive student registration |
| `python main.py` | Start real-time recognition |
| `python main.py --train` | Re-scan dataset and update encodings |
| `python main.py --summary` | Print today's log summary (No camera) |
| `python main.py --logs` | List all available log files |

---

## Example Console Output

```
[ATTENDANCE] ✓ 'Alice Smith' [CS2023001] marked PRESENT at 09:14:32.
[WARNING] Duplicate detected: 'Alice Smith' [CS2023001] is already marked present.

=================================================================
  Attendance Summary — 2026-03-25
=================================================================
  #    Name                   Reg No           Time
  ------------------------------------------------------------
  1    Alice Smith            CS2023001        09:14:32
  2    Bob Jones              CS2023002        09:15:01

  Total present: 2
=================================================================
```

---

## Edge Cases Handled

- **Unknown Face:** Labeled "Unknown" in red; attendance is NOT marked.
- **No Face:** Status text "No face in frame" appears.
- **Duplicate:** Prevents logging the same student twice in 24 hours.
- **Cooldown:** A 5-second wait before recognizing the same person again (prevents log flooding).

---

## Tech Stack

- **Python 3.9+**
- **OpenCV** — Frame processing
- **dlib / face_recognition** — AI modeling
- **NumPy** — Math & calculations
- **CSV** — Storage

---

## License
MIT — Free to use for university projects and personal use.
