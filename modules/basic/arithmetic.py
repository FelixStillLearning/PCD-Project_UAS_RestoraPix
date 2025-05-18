"""
Arithmetic operations module
Contains functions for basic arithmetic operations on images
"""
import cv2
import numpy as np

def add_images(img1, img2):
    """
    Add two images using manual operation (not cv2.add)
    
    Parameters:
    img1 (numpy.ndarray): First input image
    img2 (numpy.ndarray): Second input image
    
    Returns:
    numpy.ndarray: Sum of images
    """
    # Check if images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Perform addition 
    result = np.clip(img1.astype(np.int16) + img2.astype(np.int16), 0, 255).astype(np.uint8)
    
    return result

def subtract_images(img1, img2):
    """
    Subtract one image from another using manual operation (not cv2.subtract)
    
    Parameters:
    img1 (numpy.ndarray): First input image
    img2 (numpy.ndarray): Second input image to subtract from the first
    
    Returns:
    numpy.ndarray: Difference of images
    """
    # Check if images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Perform subtraction
    result = np.clip(img1.astype(np.int16) - img2.astype(np.int16), 0, 255).astype(np.uint8)
    
    return result

def multiply_images(img1, img2):
    """
    Multiply two images using manual operation (not cv2.multiply)
    
    Parameters:
    img1 (numpy.ndarray): First input image
    img2 (numpy.ndarray): Second input image
    
    Returns:
    numpy.ndarray: Product of images
    """
    # Check if images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Perform multiplication 
    # Scale down to avoid overflow
    result = np.clip((img1.astype(np.float32) * img2.astype(np.float32)) / 255.0, 0, 255).astype(np.uint8)
    
    return result

def divide_images(img1, img2):
    """
    Divide one image by another using manual operation (not cv2.divide)
    
    Parameters:
    img1 (numpy.ndarray): First input image (numerator)
    img2 (numpy.ndarray): Second input image (denominator)
    
    Returns:
    numpy.ndarray: Quotient of images
    """
    # Check if images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Avoid division by zero
    img2_safe = np.where(img2 == 0, 1, img2)
    
    # Perform division manually
    result = np.clip((img1.astype(np.float32) / img2_safe.astype(np.float32)) * 255.0, 0, 255).astype(np.uint8)
    
    return result

def bitwise_and(img1, img2):
    """
    Perform bitwise AND operation on two images
    
    Parameters:
    img1 (numpy.ndarray): First input image
    img2 (numpy.ndarray): Second input image
    
    Returns:
    numpy.ndarray: Result of bitwise AND operation
    """
    # Check if images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Perform bitwise AND
    result = cv2.bitwise_and(img1, img2)
    
    return result

def bitwise_or(img1, img2):
    """
    Perform bitwise OR operation on two images
    
    Parameters:
    img1 (numpy.ndarray): First input image
    img2 (numpy.ndarray): Second input image
    
    Returns:
    numpy.ndarray: Result of bitwise OR operation
    """
    # Check if images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Perform bitwise OR
    result = cv2.bitwise_or(img1, img2)
    
    return result

def bitwise_xor(img1, img2):
    """
    Perform bitwise XOR operation on two images
    
    Parameters:
    img1 (numpy.ndarray): First input image
    img2 (numpy.ndarray): Second input image
    
    Returns:
    numpy.ndarray: Result of bitwise XOR operation
    """
    # Check if images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Perform bitwise XOR
    result = cv2.bitwise_xor(img1, img2)
    
    return result

def bitwise_not(img):
    """
    Perform bitwise NOT operation on an image
    
    Parameters:
    img (numpy.ndarray): Input image
    
    Returns:
    numpy.ndarray: Result of bitwise NOT operation (inverted image)
    """
    # Perform bitwise NOT
    result = cv2.bitwise_not(img)
    
    return result