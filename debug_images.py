"""
Diagnostic: prints dtype, shape, contiguity, and min/max for every image
in the dataset before face_recognition sees it.
"""
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT_DIR    = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
SUPPORTED   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for person_dir in sorted(DATASET_DIR.iterdir()):
    if not person_dir.is_dir():
        continue
    print(f"\n=== {person_dir.name} ===")
    for img_path in person_dir.iterdir():
        if img_path.suffix.lower() not in SUPPORTED:
            continue
        try:
            img = Image.open(str(img_path))
            print(f"  PIL mode before convert : {img.mode}  size={img.size}")
            img_rgb = img.convert("RGB")
            rgb = np.array(img_rgb)
            print(f"  After PIL→numpy         : dtype={rgb.dtype}  shape={rgb.shape}  C-contig={rgb.data.c_contiguous}")

            h, w = rgb.shape[:2]
            if max(h, w) > 1600:
                scale = 1600 / max(h, w)
                rgb = cv2.resize(rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                print(f"  After cv2.resize        : dtype={rgb.dtype}  shape={rgb.shape}  C-contig={rgb.data.c_contiguous}")

            # normalize
            if rgb.ndim == 3 and rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]
            rgb = rgb.astype(np.uint8)
            rgb = np.ascontiguousarray(rgb)
            print(f"  After _normalize_for_fr : dtype={rgb.dtype}  shape={rgb.shape}  C-contig={rgb.data.c_contiguous}  min={rgb.min()}  max={rgb.max()}")

            # Now try face_recognition
            import face_recognition
            boxes = face_recognition.face_locations(rgb, model="hog")
            print(f"  face_recognition result : {len(boxes)} face(s) found  ✓")

        except Exception as e:
            print(f"  ERROR on {img_path.name}: {e}")
