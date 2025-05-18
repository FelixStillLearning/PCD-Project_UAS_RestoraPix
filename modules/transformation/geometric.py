"""
Geometric transformations module
Contains functions for geometric transformations of images
"""
import cv2
import numpy as np

def translate(image, dx=50, dy=50):
    """
    Translate an image by shifting it
    
    Parameters:
    image (numpy.ndarray): Input image
    dx (int): Horizontal shift (positive = right, negative = left)
    dy (int): Vertical shift (positive = down, negative = up)
    
    Returns:
    numpy.ndarray: Translated image
    """
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Create translation matrix
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    
    # Apply affine transformation
    translated_image = cv2.warpAffine(image, translation_matrix, (w, h))
    
    return translated_image

def rotate(image, angle=45, scale=1.0):
    """
    Rotate an image by a specified angle
    
    Parameters:
    image (numpy.ndarray): Input image
    angle (float): Rotation angle in degrees (positive = counterclockwise)
    scale (float): Scaling factor
    
    Returns:
    numpy.ndarray: Rotated image
    """
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Calculate the center of the image
    center = (w // 2, h // 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Apply affine transformation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    return rotated_image

def rotate90(image):
    """
    Rotate an image by 90 degrees counterclockwise
    
    Parameters:
    image (numpy.ndarray): Input image
    
    Returns:
    numpy.ndarray: Rotated image
    """
    return rotate(image, 90)

def rotate180(image):
    """
    Rotate an image by 180 degrees
    
    Parameters:
    image (numpy.ndarray): Input image
    
    Returns:
    numpy.ndarray: Rotated image
    """
    return rotate(image, 180)

def rotate270(image):
    """
    Rotate an image by 270 degrees counterclockwise (90 degrees clockwise)
    
    Parameters:
    image (numpy.ndarray): Input image
    
    Returns:
    numpy.ndarray: Rotated image
    """
    return rotate(image, -90)

def transpose(image):
    """
    Transpose an image (swap rows and columns)
    
    Parameters:
    image (numpy.ndarray): Input image
    
    Returns:
    numpy.ndarray: Transposed image
    """
    return cv2.transpose(image)

def zoom(image, factor=2.0):
    """
    Zoom into an image by a scaling factor
    
    Parameters:
    image (numpy.ndarray): Input image
    factor (float): Scaling factor (> 1.0 for zoom in, < 1.0 for zoom out)
    
    Returns:
    numpy.ndarray: Zoomed image
    """
    # Calculate new dimensions
    h, w = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    
    # Resize image
    zoomed_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    return zoomed_image

def zoom_in(image, factor=2.0):
    """
    Zoom in on an image
    
    Parameters:
    image (numpy.ndarray): Input image
    factor (float): Zoom factor (1.0 = original size, 2.0 = twice size)
    
    Returns:
    numpy.ndarray: Zoomed in image
    """
    if factor <= 1.0:
        raise ValueError("Zoom in factor must be greater than 1.0")
    
    return zoom(image, factor)

def zoom_out(image, factor=0.5):
    """
    Zoom out from an image
    
    Parameters:
    image (numpy.ndarray): Input image
    factor (float): Zoom factor (0.5 = half size, 0.25 = quarter size)
    
    Returns:
    numpy.ndarray: Zoomed out image
    """
    if factor >= 1.0:
        raise ValueError("Zoom out factor must be less than 1.0")
    
    return zoom(image, factor)

def skew(image, new_width=None, new_height=None):
    """
    Apply a skew transformation to an image (non-uniform scaling)
    
    Parameters:
    image (numpy.ndarray): Input image
    new_width (int): Target width (if None, use original width)
    new_height (int): Target height (if None, use original height)
    
    Returns:
    numpy.ndarray: Skewed image
    """
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Set default values if not provided
    if new_width is None:
        new_width = w
    if new_height is None:
        new_height = h
    
    # Resize image with non-uniform scaling
    skewed_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return skewed_image

def crop(image, x=0, y=0, width=None, height=None):
    """
    Crop a region from an image
    
    Parameters:
    image (numpy.ndarray): Input image
    x (int): X-coordinate of top-left corner of crop region
    y (int): Y-coordinate of top-left corner of crop region
    width (int): Width of crop region (if None, use width-x)
    height (int): Height of crop region (if None, use height-y)
    
    Returns:
    numpy.ndarray: Cropped image
    """
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Set default values if not provided
    if width is None:
        width = w - x
    if height is None:
        height = h - y
    
    # Validate crop region
    if x < 0 or y < 0 or x + width > w or y + height > h:
        raise ValueError("Crop region is outside image boundaries")
    
    # Extract the region of interest
    cropped_image = image[y:y+height, x:x+width]
    
    return cropped_image