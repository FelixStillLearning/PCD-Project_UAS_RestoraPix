#!/usr/bin/env python3
"""
Create test images with various characters for alphabetic recognition testing
"""

import cv2
import numpy as np
from pathlib import Path

def create_test_images():
    """Create test images with different character layouts"""
    
    # Create output directory
    output_dir = Path("assets/test_images")
    output_dir.mkdir(exist_ok=True)
    
    # Test Image 1: Simple text
    img1 = np.ones((300, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img1, "HELLO123", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(img1, "ABC", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4)
    cv2.imwrite(str(output_dir / "test_simple.png"), img1)
    
    # Test Image 2: Multiple lines
    img2 = np.ones((400, 500, 3), dtype=np.uint8) * 255
    texts = ["ABC123", "HELLO", "WORLD", "456"]
    for i, text in enumerate(texts):
        y_pos = 80 + i * 80
        cv2.putText(img2, text, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.imwrite(str(output_dir / "test_multiline.png"), img2)
    
    # Test Image 3: Different fonts and sizes
    img3 = np.ones((350, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img3, "BIG", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4)
    cv2.putText(img3, "medium", (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.putText(img3, "small", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img3, "Numbers: 456789", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.imwrite(str(output_dir / "test_varied.png"), img3)
    
    # Test Image 4: Black background with white text
    img4 = np.zeros((300, 500, 3), dtype=np.uint8)
    cv2.putText(img4, "WHITE TEXT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(img4, "ON BLACK", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imwrite(str(output_dir / "test_inverted.png"), img4)
    
    print(f"Test images created in {output_dir}/")
    print("Available test images:")
    for img_file in output_dir.glob("*.png"):
        print(f"  - {img_file.name}")

if __name__ == "__main__":
    create_test_images()
