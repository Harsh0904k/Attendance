# SMART ATTENDANCE SYSTEM — Face Recognition CV

> A production-grade Computer Vision system that marks attendance in real-time, prevents duplicate entries, and manages a student dataset. Optimized for **Windows 10/11** and **Python 3.12**.

---

## 📋 Table of Contents
1. [Prerequisites](#-prerequisites)
2. [Detailed Windows Setup (dlib)](#-detailed-windows-setup-dlib)
3. [Quick Start Guide](#-quick-start-guide)
4. [CLI Module Reference](#-cli-module-reference)
5. [Troubleshooting FAQ](#-troubleshooting-faq)

---

## 💻 Prerequisites

| Component | Requirement |
| :--- | :--- |
| **Python** | **3.12.x** (Standard 64-bit) |
| **NumPy** | **< 2.0.0** (MANDATORY for dlib compatibility) |
| **OpenCV** | **opencv-python** (NOT headless version) |
| **Hardware** | Webcam (Built-in or USB) |

---

## 🛠 Detailed Windows Setup (dlib)

The `dlib` engine is the most challenging part to install on Windows. Follow **ONE** of these two methods:

### Method A: Using Pre-built Wheels (Fastest)
1.  Check your Python version: `python --version`.
2.  Go to [Dlib Windows Wheels Repository](https://github.com/z-mahmud22/Dlib_Windows_Python3.x/releases).
3.  Download the `.whl` file that matches your version (e.g., `cp312` for Python 3.12).
4.  In your terminal, navigate to the download folder and run:
    ```powershell
    pip install dlib-19.24.99-cp312-cp312-win_amd64.whl
    ```

### Method B: From Source (Robust)
1.  Install **[Visual Studio Community](https://visualstudio.microsoft.com/downloads/)**.
2.  During installation, select **"Desktop development with C++"**.
3.  Ensure "C++ CMake tools for Windows" is checked in the optional components.
4.  Open the "Developer Command Prompt for VS" and run:
    ```powershell
    pip install cmake
    pip install dlib
    ```

---

## ⚡ Quick Start Guide

### 1. Initialize Environment
```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install NumPy 1.x first to avoid dlib errors
pip install "numpy<2.0.0"

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Register a Student
Use the interactive flow to add a student to the system.
```bash
python main.py --register
```
- Path to photo should be an absolute path (e.g. `C:\Users\Name\Desktop\photo.jpg`).
- The system will auto-resize and normalize the image.

### 3. Generate Encodings
You must run this at least once after registering students.
```bash
python main.py --train
```

### 4. Run Live Recognition
```bash
python main.py
```
- **Q:** Quit.
- **S:** Show today's attendance summary.

---

## 📖 CLI Module Reference

| Command | Goal | Why use it? |
| :--- | :--- | :--- |
| `python main.py --register` | Registration | Add new student info + photo safely. |
| `python main.py --train` | Training | Update the digital "brain" (`face_encodings.pkl`). |
| `python main.py` | Attendance | Start the real-time webcam scanner. |
| `python main.py --summary` | Logs | Print today's attendane without opening camera. |
| `python main.py --logs` | Audit | List all CSV log files in the `attendance_logs` directory. |

---

## ❓ Troubleshooting FAQ

### 1. `RuntimeError: Unsupported image type`
**Cause:** You have NumPy 2.x installed. dlib 19.x is built for NumPy 1.x.
**Fix:** Run `pip install "numpy<2.0.0"`.

### 2. `cv2.error: Unspecified error (...window.cpp)`
**Cause:** `opencv-python-headless` is installed instead of `opencv-python`.
**Fix:** 
```powershell
pip uninstall opencv-python-headless -y
pip install opencv-python
```

### 3. `AttributeError: module 'cv2' has no attribute 'VideoCapture'`
**Cause:** Python 3.12 eagerly evaluating type hints.
**Solution:** We have implemented a "Forward Reference" fix in `src/recognize.py`. Ensure you are using the latest code where the return type is wrapped in quotes: `-> "cv2.VideoCapture"`.

### 4. Camera won't open
**Fix:** Open `src/recognize.py` and change `CAMERA_INDEX = 0` to `1` or `2` if you have multiple cameras connected.

---

## 📂 Project Structure
- `dataset/`: Raw student images organized by `RegNo_Name`.
- `src/`: Core logic modules (Attendace, Training, Recognition).
- `attendance_logs/`: Auto-generated daily CSV reports.
- `main.py`: The single entry point for all operations.

---

## ⚖ License
MIT — Free to use for projects, portfolios, and university submissions.
