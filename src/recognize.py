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
  5. Call mark_attendance(name, regno) for every confirmed identity.
  6. Overlay bounding boxes, name, and reg. number on the display frame.
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

RESIZE_SCALE          = 0.5      # Scale frames down for speed (keep < 1.0)
RECOGNITION_THRESHOLD = 0.50     # Maximum face distance to accept a match
COOLDOWN_SECONDS      = 5        # Skip repeated recognition attempts per face
CAMERA_INDEX          = 0        # Default webcam index

# Colours (BGR)
COLOR_KNOWN   = (0, 200, 80)    # green for recognised faces
COLOR_UNKNOWN = (0, 80, 220)    # blue-red for unknown faces


def load_encodings(encodings_path: Path) -> tuple[list, list[str], list[str]]:
    """
    Load pre-computed encodings from disk.
    Returns (encodings, names, regnos). Exits if file not found.
    """
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
    # Graceful fallback for pickle files generated before reg. number support
    regnos    = data.get("regnos", ["N/A"] * len(names))

    if not encodings:
        print("[ERROR] No encodings found in the saved file. Re-run training.")
        sys.exit(1)

    unique_students = len(set(zip(names, regnos)))
    print(f"[RECOG] Loaded {len(encodings)} encoding(s) for {unique_students} student(s).")
    return encodings, names, regnos


def identify_face(
    face_encoding: np.ndarray,
    known_encodings: list,
    known_names: list[str],
    known_regnos: list[str],
) -> tuple[str, str, float]:
    """
    Compare a single face encoding against all known encodings.

    Returns:
        (name, regno, distance) — ('Unknown', 'N/A', 1.0) when no match found.
    """
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_idx  = int(np.argmin(distances))
    best_dist = float(distances[best_idx])

    if best_dist <= RECOGNITION_THRESHOLD:
        return known_names[best_idx], known_regnos[best_idx], best_dist
    return "Unknown", "N/A", best_dist


def draw_label(
    frame: np.ndarray,
    top: int, right: int, bottom: int, left: int,
    name: str,
    regno: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a bounding box, name, and registration number on the frame in place."""
    # Main bounding box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Primary label: name
    name_label = name if name == "Unknown" else f"{name}  |  {regno}"
    (text_w, text_h), _ = cv2.getTextSize(name_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    label_bg_top = bottom - text_h - 12
    cv2.rectangle(frame, (left, label_bg_top), (left + text_w + 8, bottom), color, -1)
    cv2.putText(
        frame, name_label,
        (left + 4, bottom - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 1, cv2.LINE_AA,
    )


def open_camera(index: int) -> "cv2.VideoCapture":
    """Open the webcam and verify it is accessible."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(
            f"[ERROR] Could not open camera at index {index}. "
            "Check that a webcam is connected and not in use by another app."
        )
        sys.exit(1)
    return cap


def run_recognition() -> None:
    """
    Main recognition loop.

    Press 'q' to quit, 's' to print today's attendance summary.
    """
    known_encodings, known_names, known_regnos = load_encodings(ENCODINGS_FILE)

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
            small     = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
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
                cv2.putText(
                    frame, "No face in frame",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 165, 255), 2, cv2.LINE_AA,
                )

            now = time.time()

            for (top, right, bottom, left), encoding in zip(scaled_locations, face_encodings):
                name, regno, distance = identify_face(
                    encoding, known_encodings, known_names, known_regnos
                )

                if name != "Unknown":
                    color = COLOR_KNOWN
                    # Attendance with cooldown guard
                    if now - last_seen.get(name, 0) > COOLDOWN_SECONDS:
                        mark_attendance(name, regno)
                        last_seen[name] = now
                else:
                    color = COLOR_UNKNOWN

                draw_label(frame, top, right, bottom, left, name, regno, color)

        # ── Overlay HUD ───────────────────────────────────────────────────
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


def run_recognition_from_image(image_path_str: str) -> None:
    """
    Process a single image file for attendance.
    Useful for 'uploading' a photo to mark attendance.
    """
    image_path = Path(image_path_str).resolve()
    if not image_path.exists():
        print(f"[ERROR] Image file not found: {image_path}")
        return

    known_encodings, known_names, known_regnos = load_encodings(ENCODINGS_FILE)

    try:
        rgb_frame = face_recognition.load_image_file(str(image_path))
    except Exception as e:
        print(f"[ERROR] Could not load image: {e}")
        return

    # Resize if very large (> 1600px)
    h, w = rgb_frame.shape[:2]
    if max(h, w) > 1600:
        scale = 1600 / max(h, w)
        rgb_frame = cv2.resize(rgb_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        rgb_frame = np.ascontiguousarray(rgb_frame.astype(np.uint8))

    print(f"\n[RECOG] Processing image: {image_path.name} ...")
    
    # Process the image
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not face_locations:
        print("[RECOG] No faces detected in the image.")
        return

    print(f"[RECOG] Found {len(face_locations)} face(s).")
    
    # For drawing, convert to BGR
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        name, regno, distance = identify_face(
            encoding, known_encodings, known_names, known_regnos
        )

        if name != "Unknown":
            color = COLOR_KNOWN
            mark_attendance(name, regno)
        else:
            color = COLOR_UNKNOWN
            print(f"[RECOG] Unknown face detected (distance: {distance:.2f})")

        draw_label(frame, top, right, bottom, left, name, regno, color)

    # Save and show results
    output_path = ROOT_DIR / f"attendance_result_{int(time.time())}.jpg"
    cv2.imwrite(str(output_path), frame)
    print(f"[RECOG] Result image saved to: {output_path.name}")
    
    # Show the result window
    cv2.imshow("Recognition Result (Press any key to close)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Allow running this module directly ────────────────────────────────────────
if __name__ == "__main__":
    run_recognition()
