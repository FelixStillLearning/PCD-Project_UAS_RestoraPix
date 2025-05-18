"""
Test script to verify module imports and basic functionality
"""
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import modules to test
from refactored_src.core.image_processor import ImageProcessor
from refactored_src.modules.basic.operations import grayscale, adjust_brightness, adjust_contrast
from refactored_src.modules.basic.histogram import calculate_grayscale_histogram, plot_grayscale_histogram
from refactored_src.modules.filtering.filters import mean_filter, gaussian_filter
from refactored_src.modules.edge_detection.detectors import sobel_edge_detection
from refactored_src.modules.morphology.operations import dilation, erosion
from refactored_src.modules.segmentation.thresholding import binary_threshold
from refactored_src.modules.transformation.geometric import rotate90

def test_modules():
    """Test basic functionality of modules"""
    print("Testing Image Processing Modules...")
    
    # Create a simple test image
    test_image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), -1)
    
    # Test basic operations
    print("\nTesting basic operations...")
    gray_image = grayscale(test_image)
    bright_image = adjust_brightness(gray_image, 50)
    contrast_image = adjust_contrast(gray_image, 1.5)
    
    # Test histogram operations
    print("Testing histogram operations...")
    hist = calculate_grayscale_histogram(gray_image)
    
    # Test filtering operations
    print("Testing filtering operations...")
    mean_filtered = mean_filter(gray_image, 3)
    gaussian_filtered = gaussian_filter(gray_image, 3)
    
    # Test edge detection
    print("Testing edge detection...")
    edges = sobel_edge_detection(gray_image)
    
    # Test morphology operations
    print("Testing morphology operations...")
    dilated = dilation(gray_image, 3)
    eroded = erosion(gray_image, 3)
    
    # Test thresholding
    print("Testing thresholding...")
    binary = binary_threshold(gray_image, 127)
    
    # Test transformations
    print("Testing transformations...")
    rotated = rotate90(gray_image)
    
    # Display results
    print("All operations completed successfully!")
    
    # Create a figure with subplots
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    
    # Display original and processed images
    axs[0, 0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original')
    axs[0, 1].imshow(gray_image, cmap='gray')
    axs[0, 1].set_title('Grayscale')
    axs[0, 2].imshow(bright_image, cmap='gray')
    axs[0, 2].set_title('Brightened')
    axs[1, 0].imshow(edges, cmap='gray')
    axs[1, 0].set_title('Sobel Edges')
    axs[1, 1].imshow(mean_filtered, cmap='gray')
    axs[1, 1].set_title('Mean Filter')
    axs[1, 2].imshow(binary, cmap='gray')
    axs[1, 2].set_title('Binary Threshold')
    axs[2, 0].imshow(dilated, cmap='gray')
    axs[2, 0].set_title('Dilation')
    axs[2, 1].imshow(eroded, cmap='gray')
    axs[2, 1].set_title('Erosion')
    axs[2, 2].imshow(rotated, cmap='gray')
    axs[2, 2].set_title('Rotated 90°')
    
    # Remove axes
    for ax in axs.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.show()

if __name__ == "__main__":
    test_modules()
