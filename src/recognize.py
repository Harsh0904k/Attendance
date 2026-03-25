"""
recognize.py
─────────────────────────────────────────────────────────────────────────────
Real-time face recognition using the pre-computed encodings produced by
train.py.  Recognised faces are passed to attendance.py for logging.

Algorithm:
  1. Capture frames from the default webcam.
  2. Resize each frame for faster processing.
  3. Detect all faces in the frame (HOG model).
  4. Compare each detected face against the known encodings.
  5. Call mark_attendance() for every confirmed identity.
  6. Overlay bounding boxes and labels on the display frame.
  7. Exit when 'q' is pressed or the camera becomes unavailable.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import face_recognition
import numpy as np

from src.attendance import mark_attendance, get_today_summary

# ── Configuration ──────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent
ENCODINGS_FILE = ROOT_DIR / "src" / "face_encodings.pkl"

RESIZE_SCALE      = 0.5      # Scale frames down for speed (keep < 1.0)
RECOGNITION_THRESHOLD = 0.50 # Maximum face distance to accept a match
COOLDOWN_SECONDS  = 5        # Skip repeated recognition attempts for same face
CAMERA_INDEX      = 0        # Default webcam index

# Colours (BGR)
COLOR_KNOWN   = (0, 200, 80)    # green for recognised faces
COLOR_UNKNOWN = (0, 80, 220)    # orange-red for unknown faces
COLOR_TEXT_BG = (20, 20, 20)


def load_encodings(encodings_path: Path) -> tuple[list, list[str]]:
    """Load pre-computed encodings from disk. Exit if file not found."""
    if not encodings_path.exists():
        print(
            "[ERROR] Encodings file not found. "
            "Please run training first:\n  python main.py --train"
        )
        sys.exit(1)

    with open(encodings_path, "rb") as fh:
        data = pickle.load(fh)

    encodings = data.get("encodings", [])
    names     = data.get("names", [])

    if not encodings:
        print("[ERROR] No encodings found in the saved file. Re-run training.")
        sys.exit(1)

    print(f"[RECOG] Loaded {len(encodings)} face encoding(s) for {len(set(names))} person(s).")
    return encodings, names


def identify_face(
    face_encoding: np.ndarray,
    known_encodings: list,
    known_names: list[str],
) -> tuple[str, float]:
    """
    Compare a single face encoding against all known encodings.

    Returns:
        (name, distance) — 'Unknown' and 1.0 when no match is found.
    """
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_idx  = int(np.argmin(distances))
    best_dist = float(distances[best_idx])

    if best_dist <= RECOGNITION_THRESHOLD:
        return known_names[best_idx], best_dist
    return "Unknown", best_dist


def draw_label(
    frame: np.ndarray,
    top: int, right: int, bottom: int, left: int,
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a bounding box and name label on the frame in place."""
    # Bounding box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Label background
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (left, bottom - text_h - 10), (left + text_w + 6, bottom), color, -1)

    # Label text
    cv2.putText(
        frame, label,
        (left + 3, bottom - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 1, cv2.LINE_AA,
    )


def open_camera(index: int) -> cv2.VideoCapture:
    """Open the webcam and verify it is accessible."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)   # CAP_DSHOW for faster init on Windows
    if not cap.isOpened():
        # Fallback: try without backend flag
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {index}. "
              "Check that a webcam is connected and not in use by another app.")
        sys.exit(1)
    return cap


def run_recognition() -> None:
    """
    Main recognition loop.

    Press 'q' to quit, 's' to print today's attendance summary.
    """
    known_encodings, known_names = load_encodings(ENCODINGS_FILE)

    cap = open_camera(CAMERA_INDEX)
    print("\n[RECOG] Camera opened. Press 'q' to quit, 's' for summary.")
    print("[RECOG] Scanning for faces…\n")

    # Cooldown tracker: name → last-recognised timestamp
    last_seen: dict[str, float] = {}

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN ] Failed to capture frame. Retrying…")
            time.sleep(0.1)
            continue

        frame_count += 1

        # ── Process every other frame to reduce CPU load ──────────────────
        if frame_count % 2 == 0:
            small = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            # Scale bounding boxes back to original frame size
            scale = 1.0 / RESIZE_SCALE
            scaled_locations = [
                (int(t * scale), int(r * scale), int(b * scale), int(l * scale))
                for (t, r, b, l) in face_locations
            ]

            if not face_locations:
                # Optionally display "No face detected" hint
                cv2.putText(
                    frame, "No face in frame",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 165, 255), 2, cv2.LINE_AA,
                )

            now = time.time()

            for (top, right, bottom, left), encoding in zip(scaled_locations, face_encodings):
                name, distance = identify_face(encoding, known_encodings, known_names)

                if name != "Unknown":
                    color = COLOR_KNOWN
                    label = f"{name} ({distance:.2f})"

                    # Cooldown check before marking attendance
                    if now - last_seen.get(name, 0) > COOLDOWN_SECONDS:
                        mark_attendance(name)
                        last_seen[name] = now
                else:
                    color = COLOR_UNKNOWN
                    label = f"Unknown ({distance:.2f})"

                draw_label(frame, top, right, bottom, left, label, color)

        # ── Overlay HUD info ──────────────────────────────────────────────
        cv2.putText(
            frame, "Smart Attendance System  |  Q: Quit  S: Summary",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )

        cv2.imshow("Smart Attendance System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n[RECOG] Quitting recognition loop.")
            break
        elif key == ord("s"):
            get_today_summary()

    cap.release()
    cv2.destroyAllWindows()
    get_today_summary()


# ── Allow running this module directly ────────────────────────────────────────
if __name__ == "__main__":
    run_recognition()
