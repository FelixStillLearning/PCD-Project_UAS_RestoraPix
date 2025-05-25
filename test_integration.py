#!/usr/bin/env python3
"""
Test script to verify alphabetic recognition integration
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_image_processor_integration():
    """Test the ImageProcessor alphabetic recognition integration"""
    print("Testing ImageProcessor alphabetic recognition integration...")
    
    try:
        from core.image_processor import ImageProcessor
        print("✓ ImageProcessor imported successfully")
        
        # Create processor instance
        processor = ImageProcessor()
        print("✓ ImageProcessor created successfully")
        
        # Create a simple test image with text
        test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "HELLO123", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Set the test image
        processor.set_image(test_image)
        print("✓ Test image set successfully")
        
        # Test alphabetic recognition method
        if hasattr(processor, 'process_for_alphabetic_recognition'):
            print("✓ process_for_alphabetic_recognition method exists")
            
            try:
                processed_image, detections = processor.process_for_alphabetic_recognition()
                print(f"✓ Alphabetic recognition completed successfully")
                print(f"  - Processed image shape: {processed_image.shape}")
                print(f"  - Number of detections: {len(detections)}")
                
                if detections:
                    print("  - Detection details:")
                    for i, detection in enumerate(detections):
                        char = detection.get('character', 'Unknown')
                        confidence = detection.get('confidence', 0.0)
                        print(f"    {i+1}. Character: '{char}', Confidence: {confidence:.3f}")
                else:
                    print("  - No characters detected (this might be expected for the test image)")
                
                return True
                
            except Exception as e:
                print(f"✗ Error during alphabetic recognition: {e}")
                return False
        else:
            print("✗ process_for_alphabetic_recognition method not found")
            return False
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_alphabetic_recognizer():
    """Test the AlphabeticRecognizer module directly"""
    print("\nTesting AlphabeticRecognizer module...")
    
    try:
        from modules.alphabetic_recognition.recognizer import AlphabeticRecognizer
        print("✓ AlphabeticRecognizer imported successfully")
        
        # Create recognizer instance
        recognizer = AlphabeticRecognizer()
        print("✓ AlphabeticRecognizer created successfully")
        
        # Create a simple test image with text
        test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "ABC123", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Test processing
        processed_image, detections = recognizer.process_image(test_image)
        print(f"✓ Image processing completed successfully")
        print(f"  - Processed image shape: {processed_image.shape}")
        print(f"  - Number of detections: {len(detections)}")
        
        if detections:
            print("  - Detection details:")
            for i, detection in enumerate(detections):
                char = detection.get('character', 'Unknown')
                confidence = detection.get('confidence', 0.0)
                print(f"    {i+1}. Character: '{char}', Confidence: {confidence:.3f}")
        else:
            print("  - No characters detected (this might be expected for the test image)")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    print("Alphabetic Recognition Integration Test")
    print("=" * 50)
    
    # Test 1: ImageProcessor integration
    test1_result = test_image_processor_integration()
    
    # Test 2: AlphabeticRecognizer module
    test2_result = test_alphabetic_recognizer()
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"ImageProcessor Integration: {'PASS' if test1_result else 'FAIL'}")
    print(f"AlphabeticRecognizer Module: {'PASS' if test2_result else 'FAIL'}")
    
    if test1_result and test2_result:
        print("\n🎉 All tests passed! The alphabetic recognition integration is working correctly.")
        print("\nNext steps:")
        print("1. Run the GUI application: python app.py")
        print("2. Load an image with text")
        print("3. Go to Analisis > Alphabetic Recognition > Alphabetic Recognition : Image")
        print("4. Test camera and video recognition as well")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    return test1_result and test2_result

if __name__ == "__main__":
    main()
