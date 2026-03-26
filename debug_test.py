
import os
from PIL import Image
import numpy as np
import face_recognition
import cv2

def test_image(path):
    print(f"Testing image: {path}")
    if not os.path.exists(path):
        print(f"ERROR: File not found")
        return

    try:
        # Load with PIL and resize immediately
        img = Image.open(path)
        print(f"  Original Size: {img.size}") # (W, H)
        
        # Resize BEFORE converting to numpy to save memory
        img.thumbnail((1200, 1200))
        print(f"  Resized Size:  {img.size}")
        
        img = img.convert('RGB')
        rgb = np.array(img)
        
        print("  Detecting faces...")
        locations = face_recognition.face_locations(rgb, model="hog")
        print(f"  Faces found: {len(locations)}")

    except Exception as e:
        print(f"  ERROR: {e}")

if __name__ == "__main__":
    test_image("harsh.jpg")
