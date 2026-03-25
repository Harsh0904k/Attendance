# Smart Attendance System using Face Recognition

> A production-ready Computer Vision project that marks attendance in real-time from a webcam feed, prevents duplicate entries, and logs everything to timestamped CSV files.

---

## Features

| Feature | Details |
|---|---|
| **Face Detection** | OpenCV + HOG model (via `face_recognition`) |
| **Face Recognition** | 128-d encoding comparison (dlib deep-metric learning) |
| **Dataset-based training** | One folder per person under `dataset/` |
| **Attendance logging** | Daily CSV files with name, date, time, status |
| **Duplicate detection** | Same person cannot be marked twice per calendar day |
| **Cooldown window** | 5-second gap between recognition events (prevents spam) |
| **Real-time webcam** | Live bounding boxes with name and confidence labels |
| **CLI modes** | `--train`, `--summary`, `--logs` flags |

---

## Folder Structure

```
smart-attendance-cv/
│
├── dataset/                   ← Training images (one sub-folder per person)
│   ├── person1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── person2/
│       └── img1.jpg
│
├── src/
│   ├── __init__.py
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

| Requirement | Version |
|---|---|
| Python | 3.9 – 3.11 (recommended) |
| Webcam | Any USB or built-in camera |
| OS | Windows 10/11, Ubuntu 20.04+, macOS 12+ |

> **Windows users:** `dlib` (a dependency of `face_recognition`) requires C++ build tools.  
> The easiest path is to install a pre-built wheel:
> ```bash
> pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/releases/download/v19.24.4/dlib-19.24.4-cp311-cp311-win_amd64.whl
> ```
> Choose the `.whl` that matches your Python version (`cp39`, `cp310`, `cp311`).

---

## Setup Instructions

### 1. Clone / download the project

```bash
git clone https://github.com/<your-username>/smart-attendance-cv.git
cd smart-attendance-cv
```

### 2. (Optional but recommended) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If you hit a `dlib` build error on Windows, install the pre-built wheel first (see Prerequisites above), then re-run the above command.

### 4. Add people to the dataset

Create one sub-folder per person inside `dataset/` and add **at least 3–5 clear face photos** (JPG or PNG):

```
dataset/
├── Alice/
│   ├── alice_front.jpg
│   └── alice_smile.jpg
└── Bob/
    ├── bob1.jpg
    └── bob2.jpg
```

**Tips for better accuracy:**
- Use well-lit, forward-facing photos.
- Include varied expressions / angles if possible.
- Folder name = the label that appears in attendance logs.

### 5. Train the model

```bash
python main.py --train
```

This scans `dataset/`, generates 128-d encodings for every detected face, and saves them to `src/face_encodings.pkl`.

### 6. Run the system

```bash
python main.py
```

If `face_encodings.pkl` does not exist yet, training runs automatically before recognition starts.

---

## How to Run (Quick Reference)

```bash
# Full pipeline (auto-trains if needed, then opens webcam)
python main.py

# Force re-train (add new people or update photos)
python main.py --train

# Show today's attendance summary without opening webcam
python main.py --summary

# List all saved attendance log files
python main.py --logs
```

**While the webcam window is open:**

| Key | Action |
|---|---|
| `Q` | Quit the system |
| `S` | Print today's attendance summary to console |

---

## Example Console Output

```
╔══════════════════════════════════════════════════════════╗
║      SMART ATTENDANCE SYSTEM — Face Recognition CV       ║
║           with Real-time Duplicate Detection             ║
╚══════════════════════════════════════════════════════════╝

[RECOG] Loaded 12 face encoding(s) for 3 person(s).
[RECOG] Camera opened. Press 'q' to quit, 's' for summary.
[RECOG] Scanning for faces…

[ATTENDANCE] ✓ 'Alice' marked PRESENT at 09:14:32 on 2026-03-25.
[ATTENDANCE] ✓ 'Bob'   marked PRESENT at 09:15:01 on 2026-03-25.
[WARNING] Duplicate detected: 'Alice' is already marked present for 2026-03-25. Skipping.

==================================================
  Attendance Summary — 2026-03-25
==================================================
   1. Alice                     09:14:32
   2. Bob                       09:15:01

  Total present: 2
==================================================
```

### Example CSV Log (`attendance_logs/attendance_2026-03-25.csv`)

```
Name,Date,Time,Status
Alice,2026-03-25,09:14:32,Present
Bob,2026-03-25,09:15:01,Present
```

---

## Adding New Users

1. Create a new folder under `dataset/`:
   ```
   dataset/NewPerson/
   ```
2. Copy at least 3–5 face images into that folder.
3. Re-run training:
   ```bash
   python main.py --train
   ```
4. Restart the system:
   ```bash
   python main.py
   ```

---

## Edge Cases Handled

| Scenario | Behaviour |
|---|---|
| No face in frame | Console hint + "No face in frame" overlay |
| Unknown face | Labelled **Unknown** with confidence score; no attendance logged |
| Duplicate attendance | `[WARNING]` printed; CSV not modified |
| Camera not available | Error message with troubleshooting hint; graceful exit |
| Empty dataset | Error message; exit before training fails silently |
| Corrupt / unreadable image | Warning per image; training continues with valid images |

---

## Tech Stack

- **Python 3.9+**
- **OpenCV** (`opencv-python`) — frame capture & rendering
- **face_recognition** — HOG face detection + dlib 128-d embeddings
- **NumPy** — vectorised distance calculations
- **csv / pathlib / argparse** — stdlib; no extra dependencies

---

## License

MIT — free to use, modify, and distribute.
