"""
Test Alphabetic Classifier Training Script
==========================================

Script untuk membuat data dummy dan menguji pipeline training.
"""

import cv2
import numpy as np
import os
from pathlib import Path

def create_dummy_character_images():
    """
    Membuat dummy images untuk testing training script.
    """
    print("Creating dummy character images for testing...")
    
    # Use absolute path
    base_path = Path(r"d:\Development\Proyek\Citra\Project_UAS\dataset\alphabets")
    
    # Characters to create (subset for testing)
    test_chars = ['A', 'B', 'C', '0', '1', '2']
    
    for char in test_chars:
        char_folder = base_path / char
        char_folder.mkdir(exist_ok=True)
        
        # Create 5 dummy images per character
        for i in range(5):
            # Create a 64x64 white image
            img = np.ones((64, 64, 3), dtype=np.uint8) * 255
            
            # Draw simple character representation
            if char == 'A':
                # Draw triangle-like shape for A
                cv2.line(img, (32, 10), (15, 50), (0, 0, 0), 3)
                cv2.line(img, (32, 10), (49, 50), (0, 0, 0), 3)
                cv2.line(img, (20, 35), (44, 35), (0, 0, 0), 2)
            elif char == 'B':
                # Draw B-like shape
                cv2.rectangle(img, (15, 10), (45, 30), (0, 0, 0), 2)
                cv2.rectangle(img, (15, 30), (45, 50), (0, 0, 0), 2)
                cv2.line(img, (15, 10), (15, 50), (0, 0, 0), 3)
            elif char == 'C':
                # Draw C-like shape
                cv2.ellipse(img, (32, 30), (15, 20), 0, 30, 330, (0, 0, 0), 3)
            elif char == '0':
                # Draw circle for 0
                cv2.circle(img, (32, 30), 15, (0, 0, 0), 3)
            elif char == '1':
                # Draw line for 1
                cv2.line(img, (32, 10), (32, 50), (0, 0, 0), 3)
                cv2.line(img, (25, 15), (32, 10), (0, 0, 0), 2)
            elif char == '2':
                # Draw 2-like shape
                cv2.ellipse(img, (32, 20), (12, 8), 0, 0, 180, (0, 0, 0), 2)
                cv2.line(img, (20, 28), (45, 45), (0, 0, 0), 2)
                cv2.line(img, (18, 50), (46, 50), (0, 0, 0), 2)
            
            # Add some noise variation
            noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Save image
            filename = char_folder / f"{char}_{i+1:02d}.png"
            cv2.imwrite(str(filename), img)
        
        print(f"Created 5 dummy images for character '{char}'")
    
    print(f"Dummy dataset created successfully!")
    print(f"Total characters: {len(test_chars)}")
    print(f"Total images: {len(test_chars) * 5}")

if __name__ == "__main__":
    create_dummy_character_images()
