"""
register_student.py
─────────────────────────────────────────────────────────────────────────────
Student registration module.

Accepts a student's name, registration number, and a photo path, then:
  1. Validates the image can be read and contains at least one face.
  2. Creates  dataset/<REGNO>_<Name>/  if it doesn't already exist.
  3. Copies the photo into that folder (numbered automatically).
  4. Optionally triggers re-training so the student can be recognised
     immediately without a separate --train step.

Usage (interactive):
    python main.py --register

Usage (fully scripted):
    python main.py --register --name "Alice Smith" --regno CS2023001 \
                              --image /path/to/alice.jpg --auto-train
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import face_recognition

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sanitise(text: str) -> str:
    """Remove characters that are unsafe in folder names."""
    unsafe = r'\/:*?"<>|'
    return "".join(c for c in text if c not in unsafe).strip()


def _validate_image(image_path: Path) -> None:
    """
    Raise ValueError if the image cannot be read or contains no detectable face.
    This gives early, friendly feedback before any files are written.
    """
    if not image_path.exists():
        raise ValueError(f"Image file not found: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{image_path.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"OpenCV could not read the image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    if not locations:
        raise ValueError(
            "No face detected in the provided image. "
            "Please use a clear, well-lit, forward-facing photo."
        )
    print(f"[REGISTER] ✓ Face detected in image ({len(locations)} face(s) found).")


def _prompt(label: str, current: str | None) -> str:
    """Prompt the user if *current* is None/empty, otherwise echo and return it."""
    if current:
        print(f"[REGISTER] {label}: {current}")
        return current.strip()
    while True:
        value = input(f"  Enter {label}: ").strip()
        if value:
            return value
        print(f"  ✗ {label} cannot be empty. Try again.")


def _next_image_filename(folder: Path, suffix: str) -> str:
    """Return an auto-numbered filename like img_001.jpg that doesn't clash."""
    existing = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    idx = len(existing) + 1
    return f"img_{idx:03d}{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def register_student(
    name: str | None = None,
    regno: str | None = None,
    image_path_str: str | None = None,
    auto_train: bool = False,
) -> None:
    """
    Full registration pipeline.

    Parameters
    ----------
    name           : Full name of the student (prompted if None).
    regno          : Registration / roll number (prompted if None).
    image_path_str : Absolute or relative path to the student's photo.
    auto_train     : If True, re-runs training automatically after saving.
    """

    print("\n" + "=" * 55)
    print("  Student Registration")
    print("=" * 55)

    # ── Collect inputs ────────────────────────────────────────────────────
    name  = _prompt("Student full name", name)
    regno = _prompt("Registration number", regno)

    if not image_path_str:
        image_path_str = _prompt("Path to student photo", None)

    image_path = Path(image_path_str).resolve()

    # ── Validate image ────────────────────────────────────────────────────
    try:
        _validate_image(image_path)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    # ── Build destination folder  dataset/REGNO_Name/ ─────────────────────
    safe_name  = _sanitise(name)
    safe_regno = _sanitise(regno)
    folder_name = f"{safe_regno}_{safe_name}"
    student_dir = DATASET_DIR / folder_name
    student_dir.mkdir(parents=True, exist_ok=True)

    # ── Copy image into student folder ────────────────────────────────────
    dest_filename = _next_image_filename(student_dir, image_path.suffix.lower())
    dest_path = student_dir / dest_filename
    shutil.copy2(str(image_path), str(dest_path))

    print(f"\n[REGISTER] Student registered successfully!")
    print(f"  Name            : {name}")
    print(f"  Registration No : {regno}")
    print(f"  Photo saved to  : {dest_path.relative_to(ROOT_DIR)}")
    print(f"  Dataset folder  : dataset/{folder_name}/")

    # ── Optional auto-train ───────────────────────────────────────────────
    if not auto_train:
        answer = input("\n  Run training now to activate this student? [Y/n]: ").strip().lower()
        auto_train = answer in ("", "y", "yes")

    if auto_train:
        print()
        # Import here to avoid circular imports during module load
        from src.train import run_training
        run_training()
    else:
        print("\n[INFO] Run  python main.py --train  to activate the new student.")

    print()


def run_registration(
    name: str | None = None,
    regno: str | None = None,
    image: str | None = None,
    auto_train: bool = False,
) -> None:
    """Thin wrapper called by main.py."""
    register_student(name, regno, image, auto_train)


# ── Allow running this module directly ────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a new student in the attendance system.")
    parser.add_argument("--name",       help="Student full name")
    parser.add_argument("--regno",      help="Registration / roll number")
    parser.add_argument("--image",      help="Path to the student's face photo")
    parser.add_argument("--auto-train", action="store_true",
                        help="Automatically re-train after registration")
    args = parser.parse_args()
    run_registration(args.name, args.regno, args.image, args.auto_train)
