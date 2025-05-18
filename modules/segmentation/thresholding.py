"""
Thresholding module
Contains functions for image segmentation through thresholding
"""
import cv2
import numpy as np
from ..basic.operations import grayscale, binarize

def binary_threshold(image, threshold=127):
    """
    Apply binary thresholding to an image
    
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
      # Apply binary threshold
    binary = binarize(gray_image, threshold)
    
    return binary

def binary_inv_threshold(image, threshold=127):
    """
    Apply inverted binary thresholding to an image
    
    Parameters:
    image (numpy.ndarray): Input image
    threshold (int): Threshold value (0-255)
    
    Returns:
    numpy.ndarray: Inverted binary image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = grayscale(image)
    else:
        gray_image = image.copy()
    
    # Apply inverted binary threshold
    _, binary_inv = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)
    
    return binary_inv

def trunc_threshold(image, threshold=127):
    """
    Apply truncated thresholding to an image
    
    Parameters:
    image (numpy.ndarray): Input image
    threshold (int): Threshold value (0-255)
    
    Returns:
    numpy.ndarray: Truncated image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = grayscale(image)
    else:
        gray_image = image.copy()
    
    # Apply truncated threshold
    _, trunc = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_TRUNC)
    
    return trunc

def tozero_threshold(image, threshold=127):
    """
    Apply to-zero thresholding to an image
    
    Parameters:
    image (numpy.ndarray): Input image
    threshold (int): Threshold value (0-255)
    
    Returns:
    numpy.ndarray: To-zero thresholded image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = grayscale(image)
    else:
        gray_image = image.copy()
    
    # Apply to-zero threshold
    _, to_zero = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_TOZERO)
    
    return to_zero

def tozero_inv_threshold(image, threshold=127):
    """
    Apply inverted to-zero thresholding to an image
    
    Parameters:
    image (numpy.ndarray): Input image
    threshold (int): Threshold value (0-255)
    
    Returns:
    numpy.ndarray: Inverted to-zero thresholded image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = grayscale(image)
    else:
        gray_image = image.copy()
    
    # Apply inverted to-zero threshold
    _, to_zero_inv = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_TOZERO_INV)
    
    return to_zero_inv

def adaptive_mean_threshold(image, block_size=11, c=2):
    """
    Apply adaptive mean thresholding to an image
    
    Parameters:
    image (numpy.ndarray): Input image
    block_size (int): Size of the pixel neighborhood for thresholding
    c (int): Constant subtracted from the mean
    
    Returns:
    numpy.ndarray: Adaptive thresholded image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = grayscale(image)
    else:
        gray_image = image.copy()
    
    # Apply adaptive mean thresholding
    result = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
    )
    
    return result

def adaptive_gaussian_threshold(image, block_size=11, c=2):
    """
    Apply adaptive Gaussian thresholding to an image
    
    Parameters:
    image (numpy.ndarray): Input image
    block_size (int): Size of the pixel neighborhood for thresholding
    c (int): Constant subtracted from the weighted mean
    
    Returns:
    numpy.ndarray: Adaptive thresholded image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = grayscale(image)
    else:
        gray_image = image.copy()
    
    # Apply adaptive Gaussian thresholding
    result = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
    )
    
    return result

def otsu_threshold(image):
    """
    Apply Otsu's thresholding to an image
    
    Parameters:
    image (numpy.ndarray): Input image
    
    Returns:
    tuple: (thresholded image, threshold value)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = grayscale(image)
    else:
        gray_image = image.copy()
    
    # Apply Otsu's thresholding
    ret, result = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return result, ret

def contour_detection(image, color_shapes=True):
    """
    Detect contours and identify shapes in an image
    
    Parameters:
    image (numpy.ndarray): Input image
    color_shapes (bool): Whether to color the detected shapes
    
    Returns:
    tuple: (image with contours, detected contours, shape types)
    """
    # Convert to RGB for visualization if needed
    if len(image.shape) == 2:
        display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        display_image = image.copy()
    
    # Convert to grayscale for processing
    if len(image.shape) == 3:
        gray_image = grayscale(image)
    else:
        gray_image = image.copy()
    
    # Apply threshold
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create white background for colored shapes if requested
    height, width = display_image.shape[:2]
    result_image = display_image.copy()
    if color_shapes:
        result_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Filter out very small contours and the outer rectangle
    filtered_contours = []
    shapes = []
    img_area = height * width
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter small contours and contours that are too large
        if 100 < area < (img_area * 0.95):
            filtered_contours.append(cnt)
            
            # Get approximate polygon
            epsilon = 0.04 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Calculate center of contour
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = 0, 0
            
            # Identify shape based on number of vertices
            num_vertices = len(approx)
            shape = "Unknown"
            shape_color = (0, 0, 0)  # Default black
            text_color = (255, 255, 255)  # Default white text
            
            if num_vertices == 3:
                shape = "Triangle"
                shape_color = (0, 0, 255)  # Red
                text_color = (0, 0, 0)
            elif num_vertices == 4:
                # Check if it's a square or rectangle
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.95 <= aspect_ratio <= 1.05:
                    shape = "Square"
                    shape_color = (0, 255, 0)  # Green
                    text_color = (0, 0, 0)
                else:
                    shape = "Rectangle"
                    shape_color = (255, 0, 0)  # Blue
                    text_color = (0, 0, 0)
            elif num_vertices == 10 or num_vertices == 5:
                shape = "Star"
                shape_color = (255, 255, 0)  # Cyan
                text_color = (0, 0, 0)
            else:
                # For circles - check circularity
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                if circularity > 0.7:
                    shape = "Circle"
                    shape_color = (0, 255, 255)  # Yellow
                    text_color = (0, 0, 0)
            
            shapes.append({"shape": shape, "center": (cx, cy), "contour": cnt})
            
            if color_shapes:
                # Fill the shape with its color
                cv2.drawContours(result_image, [cnt], -1, shape_color, -1)
                # Draw the outline
                cv2.drawContours(result_image, [cnt], -1, (0, 0, 0), 1)
                # Add text label at center
                cv2.putText(result_image, shape, (cx-20, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            else:
                # Just draw contours on the original image
                cv2.drawContours(result_image, [cnt], -1, (0, 255, 0), 2)
                cv2.putText(result_image, shape, (cx-20, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return result_image, filtered_contours, shapes