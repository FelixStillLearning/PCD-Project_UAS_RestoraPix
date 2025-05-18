"""
Basic image operations module
Contains functions for fundamental image processing operations like 
grayscale conversion, brightness, contrast, etc.
"""
import numpy as np
import cv2

def grayscale(image):
    """
    Convert an RGB image to grayscale using weighted method
    
    Parameters:
    image (numpy.ndarray): Input RGB image
    
    Returns:
    numpy.ndarray: Grayscale image
    """
    if image is None:
        return None
        
    # Return image if already grayscale
    if len(image.shape) == 2:
        return image
        
    H, W = image.shape[:2]
    gray = np.zeros((H, W), np.uint8)

    for i in range(H):
        for j in range(W):
            gray[i, j] = np.clip(0.299 * image[i, j, 2] +
                                0.587 * image[i, j, 1] +
                                0.114 * image[i, j, 0], 0, 255)

    return gray

def adjust_brightness(image, value=80):
    """
    Adjust the brightness of an image
    
    Parameters:
    image (numpy.ndarray): Input grayscale image
    value (int): Brightness adjustment value (-255 to 255)
    
    Returns:
    numpy.ndarray: Brightness adjusted image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    H, W = gray_image.shape[:2]
    result = np.zeros((H, W), np.uint8)
    
    for i in range(H):
        for j in range(W):
            pixel = gray_image[i, j]
            new_pixel = np.clip(pixel + value, 0, 255)
            result[i, j] = new_pixel

    return result

def adjust_contrast(image, factor=1.7):
    """
    Adjust the contrast of an image
    
    Parameters:
    image (numpy.ndarray): Input grayscale image
    factor (float): Contrast adjustment factor (0.0 to 3.0)
    
    Returns:
    numpy.ndarray: Contrast adjusted image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    H, W = gray_image.shape[:2]
    result = np.zeros((H, W), np.uint8)
    
    for i in range(H):
        for j in range(W):
            pixel = gray_image[i, j]
            new_pixel = np.clip(pixel * factor, 0, 255)
            result[i, j] = new_pixel

    return result

def contrast_stretching(image):
    """
    Perform contrast stretching on an image
    
    Parameters:
    image (numpy.ndarray): Input grayscale image
    
    Returns:
    numpy.ndarray: Contrast stretched image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    H, W = gray_image.shape[:2]
    result = np.zeros((H, W), np.uint8)
    
    # Find minimum and maximum values in the image
    minv = np.min(gray_image)
    maxv = np.max(gray_image)
    
    # Avoid division by zero
    if maxv == minv:
        return gray_image
    
    for i in range(H):
        for j in range(W):
            pixel = gray_image[i, j]
            # Normalize the pixel value to 0-255 range
            new_pixel = int(float(pixel - minv) / (maxv - minv) * 255)
            result[i, j] = new_pixel

    return result

def negative(image):
    """
    Create a negative of an image
    
    Parameters:
    image (numpy.ndarray): Input grayscale image
    
    Returns:
    numpy.ndarray: Negative image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    H, W = gray_image.shape[:2]
    result = np.zeros((H, W), np.uint8)
    
    for i in range(H):
        for j in range(W):
            pixel = gray_image[i, j]
            result[i, j] = 255 - pixel

    return result

def binarize(image, threshold=180):
    """
    Convert an image to binary using thresholding
    
    Parameters:
    image (numpy.ndarray): Input grayscale image
    threshold (int): Threshold value (0-255)
    
    Returns:
    numpy.ndarray: Binary image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    H, W = gray_image.shape[:2]
    result = np.zeros((H, W), np.uint8)
    
    for i in range(H):
        for j in range(W):
            pixel = gray_image[i, j]
            if pixel == threshold:
                result[i, j] = 0
            elif pixel < threshold:
                result[i, j] = 1
            else:  # pixel > threshold
                result[i, j] = 255

    return result

def histogram_equalization(image):
    """
    Perform histogram equalization on an image
    
    Parameters:
    image (numpy.ndarray): Input grayscale image
    
    Returns:
    numpy.ndarray: Histogram equalized image
    tuple: (equalized image, CDF, histogram)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    # Calculate histogram
    hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])
    
    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()
    
    # Normalize CDF
    cdf_normalized = cdf * hist.max() / cdf.max()
    
    # Create mask of non-zero CDF values
    cdf_m = np.ma.masked_equal(cdf, 0)
    
    # Normalize to 0-255 range
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    
    # Fill back the masked values
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # Map the original image through the CDF
    result = cdf[gray_image]
    
    return result, cdf_normalized, hist
