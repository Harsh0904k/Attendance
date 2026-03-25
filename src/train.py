"""
train.py
─────────────────────────────────────────────────────────────────────────────
Encodes all face images found in the dataset directory and persists the
encodings to a pickle file so they can be loaded at recognition time.

Dataset layout expected:
    dataset/
        Alice/
            img1.jpg
            img2.png
            ...
        Bob/
            img1.jpg
            ...

Each sub-folder name becomes the person's identity label.
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


def load_images_from_dataset(dataset_path: Path) -> tuple[list, list[str]]:
    """
    Walk the dataset directory and collect (encoding, name) pairs.

    Returns:
        known_encodings : list of 128-d face encoding arrays
        known_names     : list of corresponding identity strings
    """
    known_encodings: list = []
    known_names: list[str] = []

    persons = [d for d in dataset_path.iterdir() if d.is_dir()]
    if not persons:
        print("[ERROR] No person folders found inside 'dataset/'. "
              "Please add images before training.")
        sys.exit(1)

    print(f"[TRAIN] Found {len(persons)} person(s) in dataset.")

    for person_dir in sorted(persons):
        name = person_dir.name
        image_files = [
            f for f in person_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not image_files:
            print(f"[WARN ] No supported images found for '{name}'. Skipping.")
            continue

        person_encodings_count = 0

        for img_path in image_files:
            # Load image via OpenCV (handles paths with non-ASCII chars)
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"[WARN ] Could not read image: {img_path.name}. Skipping.")
                continue

            # face_recognition expects RGB
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Detect faces and generate encodings
            boxes = face_recognition.face_locations(rgb, model="hog")
            if not boxes:
                print(f"[WARN ] No face detected in {img_path.name} ({name}). Skipping.")
                continue

            encodings = face_recognition.face_encodings(rgb, boxes)
            # Use the first/largest face detected in the image
            known_encodings.append(encodings[0])
            known_names.append(name)
            person_encodings_count += 1

        print(f"[TRAIN] {name:<20} → {person_encodings_count} encoding(s) added.")

    return known_encodings, known_names


def save_encodings(encodings: list, names: list[str], output_path: Path) -> None:
    """Serialize the encodings and names to a pickle file."""
    data = {"encodings": encodings, "names": names}
    with open(output_path, "wb") as fh:
        pickle.dump(data, fh)
    print(f"\n[TRAIN] Encodings saved to: {output_path.name}")
    print(f"[TRAIN] Total face encodings stored: {len(encodings)}")


def run_training() -> None:
    """
    Full training pipeline:
      1. Validate that the dataset directory exists.
      2. Load and encode all images.
      3. Persist encodings to disk.
    """
    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        print("  Create 'dataset/<person_name>/' folders and add face images.")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("  Face Recognition — Training Phase")
    print("=" * 50)

    encodings, names = load_images_from_dataset(DATASET_DIR)

    if not encodings:
        print("[ERROR] No valid face encodings were generated. "
              "Check your dataset images and try again.")
        sys.exit(1)

    save_encodings(encodings, names, ENCODINGS_FILE)
    print("[TRAIN] Training complete. Run 'python main.py' to start recognition.\n")


# ── Allow running this module directly ────────────────────────────────────────
if __name__ == "__main__":
    run_training()
