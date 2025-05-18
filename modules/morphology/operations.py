"""
Morphology module
Contains functions for morphological operations on binary images
"""
import cv2
import numpy as np
from ..basic.operations import grayscale, binarize

def to_binary(image, threshold=127):
    """
    Convert an image to binary
    
    Parameters:
    image (numpy.ndarray): Input image
    threshold (int): Threshold value (0-255)
    
    Returns:
    numpy.ndarray: Binary image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = grayscale(image)
    else:
        gray_image = image.copy()
    
    # Apply threshold
    _, binary_image = binarize(gray_image, threshold)
    
    return binary_image

def dilation(image, kernel_size=5, iterations=1):
    """
    Perform dilation on a binary image
    
    Parameters:
    image (numpy.ndarray): Input binary image
    kernel_size (int): Size of the structuring element
    iterations (int): Number of times to apply the operation
    
    Returns:
    numpy.ndarray: Dilated image
    """
    # Convert to binary if needed
    binary_image = to_binary(image)
    
    # Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    
    # Apply dilation
    dilated_image = cv2.dilate(binary_image, kernel, iterations=iterations)
    
    return dilated_image

def erosion(image, kernel_size=5, iterations=1):
    """
    Perform erosion on a binary image
    
    Parameters:
    image (numpy.ndarray): Input binary image
    kernel_size (int): Size of the structuring element
    iterations (int): Number of times to apply the operation
    
    Returns:
    numpy.ndarray: Eroded image
    """
    # Convert to binary if needed
    binary_image = to_binary(image)
    
    # Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    
    # Apply erosion
    eroded_image = cv2.erode(binary_image, kernel, iterations=iterations)
    
    return eroded_image

def opening(image, kernel_size=5):
    """
    Perform opening (erosion followed by dilation) on a binary image
    
    Parameters:
    image (numpy.ndarray): Input binary image
    kernel_size (int): Size of the structuring element
    
    Returns:
    numpy.ndarray: Opened image
    """
    # Convert to binary if needed
    binary_image = to_binary(image)
    
    # Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    
    # Apply opening
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    
    return opened_image

def closing(image, kernel_size=5):
    """
    Perform closing (dilation followed by erosion) on a binary image
    
    Parameters:
    image (numpy.ndarray): Input binary image
    kernel_size (int): Size of the structuring element
    
    Returns:
    numpy.ndarray: Closed image
    """
    # Convert to binary if needed
    binary_image = to_binary(image)
    
    # Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    
    # Apply closing
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    return closed_image

def skeletonize(image):
    """
    Perform skeletonization on a binary image
    
    Parameters:
    image (numpy.ndarray): Input binary image
    
    Returns:
    numpy.ndarray: Skeletonized image
    """
    # Convert to binary if needed
    binary_image = to_binary(image)
    
    # Initialize skeleton image
    skeleton = np.zeros(binary_image.shape, np.uint8)
    
    # Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    # Copy input image
    img = binary_image.copy()
    
    # Iterative process for skeletonization
    while True:
        # Step 1: Erode the image
        eroded = cv2.erode(img, kernel)
        
        # Step 2: Dilate the eroded image
        temp = cv2.dilate(eroded, kernel)
        
        # Step 3: Subtract the dilated image from the original
        temp = cv2.subtract(img, temp)
        
        # Step 4: Add the difference to the skeleton
        skeleton = cv2.bitwise_or(skeleton, temp)
        
        # Step 5: Update the image for next iteration
        img = eroded.copy()
        
        # Step 6: Check if there are any white pixels left
        if cv2.countNonZero(img) == 0:
            break
    
    return skeleton