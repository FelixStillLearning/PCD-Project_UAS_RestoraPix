"""
Filtering module
Contains functions for image filtering operations
"""
import numpy as np
import cv2

def convolution(img, kernel):
    """
    Perform convolution operation on an image
    
    Parameters:
    img (numpy.ndarray): Input image
    kernel (numpy.ndarray): Convolution kernel
    
    Returns:
    numpy.ndarray: Filtered image
    """
    # Get dimensions of image and kernel
    img_height, img_width = img.shape[:2]
    k_height, k_width = kernel.shape
    
    # Calculate padding needed
    pad_h = k_height // 2
    pad_w = k_width // 2
    
    # Create output image with same dimensions as input
    output = np.zeros_like(img, dtype=np.float32)
    
    # Create padded image
    padded_img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    
    # Perform convolution
    for i in range(img_height):
        for j in range(img_width):
            # Extract region of same size as kernel
            region = padded_img[i:i+k_height, j:j+k_width]
            # Perform convolution operation (dot product and sum)
            output[i, j] = np.sum(region * kernel)
    
    # Normalize pixel values
    output = np.clip(output, 0, 255)
    
    return output.astype(np.uint8)

def mean_filter(img, filter_size=3):
    """
    Apply mean filter to an image
    
    Parameters:
    img (numpy.ndarray): Input image
    filter_size (int): Size of the filter kernel (3, 5, 7, etc.)
    
    Returns:
    numpy.ndarray: Filtered image
    """
    # Create mean filter kernel
    kernel = np.ones((filter_size, filter_size), np.float32) / (filter_size * filter_size)
    
    # Apply convolution with mean filter kernel
    return convolution(img, kernel)

def gaussian_filter(img, filter_size=3):
    """
    Apply Gaussian filter to an image
    
    Parameters:
    img (numpy.ndarray): Input image
    filter_size (int): Size of the filter kernel (3 or 5)
    
    Returns:
    numpy.ndarray: Filtered image
    """
    if filter_size == 3:
        # 3x3 Gaussian kernel
        kernel = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32) / 16
    elif filter_size == 5:
        # 5x5 Gaussian kernel
        kernel = np.array([
            [1, 4, 7, 4, 1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1, 4, 7, 4, 1]
        ], dtype=np.float32) / 273
    else:
        raise ValueError("Unsupported filter size. Use 3 or 5.")
    
    # Apply convolution with Gaussian filter kernel
    return convolution(img, kernel)

def median_filter(img, filter_size=3):
    """
    Apply median filter to an image
    
    Parameters:
    img (numpy.ndarray): Input image
    filter_size (int): Size of the filter kernel (3, 5, 7, etc.)
    
    Returns:
    numpy.ndarray: Filtered image
    """
    # Ensure grayscale image
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()
    
    # Create output image
    img_out = gray_img.copy()
    
    # Get image dimensions
    h, w = gray_img.shape
    
    # Calculate filter offset
    offset = filter_size // 2
    
    # Apply median filter
    for i in range(offset, h - offset):
        for j in range(offset, w - offset):
            # Collect neighborhood values
            neighbors = []
            for k in range(-offset, offset + 1):
                for l in range(-offset, offset + 1):
                    neighbors.append(gray_img[i + k, j + l])
            
            # Sort neighbors and get median value
            neighbors.sort()
            median = neighbors[len(neighbors) // 2]
            
            # Set output pixel to median value
            img_out[i, j] = median
    
    return img_out

def max_filter(img, filter_size=3):
    """
    Apply max filter to an image
    
    Parameters:
    img (numpy.ndarray): Input image
    filter_size (int): Size of the filter kernel (3, 5, 7, etc.)
    
    Returns:
    numpy.ndarray: Filtered image
    """
    # Ensure grayscale image
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()
    
    # Create output image
    img_out = gray_img.copy()
    
    # Get image dimensions
    h, w = gray_img.shape
    
    # Calculate filter offset
    offset = filter_size // 2
    
    # Apply max filter
    for i in range(offset, h - offset):
        for j in range(offset, w - offset):
            # Collect neighborhood values
            neighbors = []
            for k in range(-offset, offset + 1):
                for l in range(-offset, offset + 1):
                    neighbors.append(gray_img[i + k, j + l])
            
            # Get maximum value
            max_value = max(neighbors)
            
            # Set output pixel to maximum value
            img_out[i, j] = max_value
    
    return img_out

def min_filter(img, filter_size=3):
    """
    Apply min filter to an image
    
    Parameters:
    img (numpy.ndarray): Input image
    filter_size (int): Size of the filter kernel (3, 5, 7, etc.)
    
    Returns:
    numpy.ndarray: Filtered image
    """
    # Ensure grayscale image
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()
    
    # Create output image
    img_out = gray_img.copy()
    
    # Get image dimensions
    h, w = gray_img.shape
    
    # Calculate filter offset
    offset = filter_size // 2
    
    # Apply min filter
    for i in range(offset, h - offset):
        for j in range(offset, w - offset):
            # Collect neighborhood values
            neighbors = []
            for k in range(-offset, offset + 1):
                for l in range(-offset, offset + 1):
                    neighbors.append(gray_img[i + k, j + l])
            
            # Get minimum value
            min_value = min(neighbors)
            
            # Set output pixel to minimum value
            img_out[i, j] = min_value
    
    return img_out
