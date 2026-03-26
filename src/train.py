"""
train.py
─────────────────────────────────────────────────────────────────────────────
Encodes all face images found in the dataset directory and persists the
encodings to a pickle file so they can be loaded at recognition time.

Dataset layout expected:
    dataset/
        REGNO_FullName/        ← folder name format: RegNo_Name
            img1.jpg
            img2.png
            ...

The folder name is split on the FIRST underscore:
    "CS2023001_Alice Smith"  →  regno="CS2023001", name="Alice Smith"

Folders without an underscore are treated as name-only (regno="N/A").
─────────────────────────────────────────────────────────────────────────────
"""

import os
import pickle
import sys
from pathlib import Path

import cv2
import face_recognition

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
ENCODINGS_FILE = ROOT_DIR / "src" / "face_encodings.pkl"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalize_for_fr(arr: "np.ndarray") -> "np.ndarray":
    """
    Guarantee the array is a uint8, 3-channel (H×W×3), C-contiguous RGB image
    — exactly what dlib / face_recognition requires.
    """
    import numpy as np
    # Drop alpha channel if present (RGBA → RGB)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    # Ensure exactly 3 dims and 3 channels
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unexpected image shape: {arr.shape}")
    # Cast to uint8 (cv2.resize can silently promote to float)
    arr = arr.astype(np.uint8)
    # Re-enforce C-contiguous memory layout
    arr = np.ascontiguousarray(arr)
    return arr


def _parse_folder_name(folder_name: str) -> tuple[str, str]:
    """
    Split <REGNO>_<Name> into (regno, name).
    Falls back to ("N/A", folder_name) when no underscore is present
    (backward-compatible with old plain-name folders).
    """
    if "_" in folder_name:
        regno, name = folder_name.split("_", 1)
        return regno.strip(), name.strip()
    return "N/A", folder_name.strip()


def load_images_from_dataset(dataset_path: Path) -> tuple[list, list[str], list[str]]:
    """
    Walk the dataset directory and collect (encoding, name, regno) triples.

    Returns:
        known_encodings : list of 128-d face encoding arrays
        known_names     : list of identity strings
        known_regnos    : list of registration numbers (parallel to names)
    """
    known_encodings: list = []
    known_names: list[str] = []
    known_regnos: list[str] = []

    persons = [d for d in dataset_path.iterdir() if d.is_dir()]
    if not persons:
        print("[ERROR] No person folders found inside 'dataset/'. "
              "Register students first:  python main.py --register")
        sys.exit(1)

    print(f"[TRAIN] Found {len(persons)} person folder(s) in dataset.")

    for person_dir in sorted(persons):
        regno, name = _parse_folder_name(person_dir.name)
        image_files = [
            f for f in person_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not image_files:
            print(f"[WARN ] No supported images for '{person_dir.name}'. Skipping.")
            continue

        person_count = 0

        for img_path in image_files:
            # Use face_recognition's own loader — only method guaranteed to be
            # dlib-compatible (bypasses numpy ABI issues on Python 3.12)
            try:
                rgb = face_recognition.load_image_file(str(img_path))
            except Exception:
                print(f"[WARN ] Could not load image: {img_path.name}. Skipping.")
                continue

            # Resize if very large (> 1600px)
            h, w = rgb.shape[:2]
            if max(h, w) > 1600:
                scale = 1600 / max(h, w)
                rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                # cv2.resize can change dtype; restore dlib-compatible uint8 C-array
                rgb = _normalize_for_fr(rgb)

            boxes = face_recognition.face_locations(rgb, model="hog")
            if not boxes:
                print(f"[WARN ] No face detected in {img_path.name} ({name}). Skipping.")
                continue

            encodings = face_recognition.face_encodings(rgb, boxes)
            known_encodings.append(encodings[0])
            known_names.append(name)
            known_regnos.append(regno)
            person_count += 1

        tag = f"[{regno}]" if regno != "N/A" else ""
        print(f"[TRAIN] {name:<22} {tag:<14} → {person_count} encoding(s) added.")

    return known_encodings, known_names, known_regnos


def save_encodings(
    encodings: list,
    names: list[str],
    regnos: list[str],
    output_path: Path,
) -> None:
    """Serialize encodings, names, and registration numbers to a pickle file."""
    data = {"encodings": encodings, "names": names, "regnos": regnos}
    with open(output_path, "wb") as fh:
        pickle.dump(data, fh)
    print(f"\n[TRAIN] Encodings saved → {output_path.name}")
    print(f"[TRAIN] Total encodings: {len(encodings)} across {len(set(names))} student(s).")


def run_training() -> None:
    """
    Full training pipeline:
      1. Validate dataset directory exists.
      2. Load and encode all face images.
      3. Persist encodings + metadata to disk.
    """
    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        print("  Run:  python main.py --register  to add students.")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  Face Recognition — Training Phase")
    print("=" * 55)

    encodings, names, regnos = load_images_from_dataset(DATASET_DIR)

    if not encodings:
        print("[ERROR] No valid face encodings generated. "
              "Check your dataset images and try again.")
        sys.exit(1)

    save_encodings(encodings, names, regnos, ENCODINGS_FILE)
    print("[TRAIN] Training complete.\n")


# ── Allow running this module directly ────────────────────────────────────────
if __name__ == "__main__":
    run_training()
