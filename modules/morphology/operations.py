"""
Morphology module
Contains functions for morphological operations on binary images
"""
import cv2
import numpy as np
from ..basic.operations import grayscale, binarize

def to_binary(image, threshold=127, use_adaptive=False, use_otsu=False):
    """
    Convert an image to binary
    
    Parameters:
    image (numpy.ndarray): Input image
    threshold (int): Threshold value (0-255)
    use_adaptive (bool): Whether to use adaptive thresholding
    use_otsu (bool): Whether to use Otsu's method for thresholding
    
    Returns:
    numpy.ndarray: Binary image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = grayscale(image)
    else:
        gray_image = image.copy()
    
    # Check if image might be inverted (more white than black)
    white_count = cv2.countNonZero(gray_image)
    total_pixels = gray_image.shape[0] * gray_image.shape[1]
    
    # Apply thresholding based on parameters
    if use_otsu:
        # Otsu's method automatically determines optimal threshold
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif use_adaptive:
        # Adaptive thresholding for images with varying lighting
        binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
    else:
        # Standard binary thresholding
        binary_image = binarize(gray_image, threshold)
    
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
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = grayscale(image)
    else:
        gray_image = image.copy()
    
    # Check if image is inverted (more white than black)
    white_count = cv2.countNonZero(gray_image)
    total_pixels = gray_image.shape[0] * gray_image.shape[1]
    
    # If image is mostly white (>90%), it might be inverted, so invert it
    if white_count > 0.9 * total_pixels:
        gray_image = cv2.bitwise_not(gray_image)
    
    # Apply threshold with OTSU to better separate foreground and background
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Check if binary image is all white or all black
    non_zero = cv2.countNonZero(binary_image)
    if non_zero == 0 or non_zero == total_pixels:
        # If all white or all black, we can't skeletonize
        return binary_image
    
    # Initialize skeleton image
    skeleton = np.zeros(binary_image.shape, np.uint8)
    
    # Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    # Copy input image
    img = binary_image.copy()
    
    # Set maximum iterations to prevent infinite loop
    max_iterations = 1000
    iter_count = 0
    
    # Iterative process for skeletonization
    while iter_count < max_iterations:
        # Step 1: Erode the image
        eroded = cv2.erode(img, kernel)
        
        # Check if erosion had any effect
        if cv2.countNonZero(eroded) == cv2.countNonZero(img):
            # If erosion doesn't change anything in 2 consecutive iterations, break
            if iter_count > 0:
                break
        
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
            
        iter_count += 1
    
    return skeleton