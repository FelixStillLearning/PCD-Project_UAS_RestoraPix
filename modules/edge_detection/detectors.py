"""
Edge detection module
Contains functions for detecting edges in images
"""
import numpy as np
import cv2
from math import sqrt
from ..filtering.filters import convolution
from ..basic.operations import grayscale

def sobel_edge_detection(img):
    """
    Detect edges using Sobel operator
    
    Parameters:
    img (numpy.ndarray): Input image
    
    Returns:
    numpy.ndarray: Edge map
    """
    # Ensure grayscale image
    if len(img.shape) == 3:
        gray_img = grayscale(img)
    else:
        gray_img = img.copy()
    
    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Apply convolution with Sobel kernels
    # gradient_x= convolution(gray_img, sobel_x)
    # gradient_y= convolution(gray_img, sobel_y)
    gradient_x = cv2.filter2D(gray_img, cv2.CV_64F, sobel_x)
    gradient_y = cv2.filter2D(gray_img, cv2.CV_64F, sobel_y)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Normalize to range 0-255
    max_magnitude = np.max(gradient_magnitude)
    if max_magnitude > 0:
        gradient_magnitude = (gradient_magnitude / max_magnitude) * 255
    
    return gradient_magnitude.astype(np.uint8)

def prewitt_edge_detection(img):
    """
    Detect edges using Prewitt operator
    
    Parameters:
    img (numpy.ndarray): Input image
    
    Returns:
    numpy.ndarray: Edge map
    """
    # Ensure grayscale image
    if len(img.shape) == 3:
        gray_img = grayscale(img)
    else:
        gray_img = img.copy()
    
    # Define Prewitt kernels
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    # Apply convolution with Prewitt kernels
    # gradient_x = convolution(gray_img, prewitt_x)
    # gradient_y = convolution(gray_img, prewitt_y)

    gradient_x = cv2.filter2D(gray_img, cv2.CV_64F, prewitt_x)
    gradient_y = cv2.filter2D(gray_img, cv2.CV_64F, prewitt_y)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Normalize to range 0-255
    max_magnitude = np.max(gradient_magnitude)
    if max_magnitude > 0:
        gradient_magnitude = (gradient_magnitude / max_magnitude) * 255
    
    return gradient_magnitude.astype(np.uint8)

def roberts_edge_detection(img):
    """
    Detect edges using Roberts operator
    
    Parameters:
    img (numpy.ndarray): Input image
    
    Returns:
    numpy.ndarray: Edge map
    """
    # Ensure grayscale image
    if len(img.shape) == 3:
        gray_img = grayscale(img)
    else:
        gray_img = img.copy()
    
    # Define Roberts kernels
    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])
    
    # Get image dimensions
    h_img, w_img = gray_img.shape
    result = np.zeros_like(gray_img, dtype=float)
    
    # Apply Roberts operator
    for i in range(h_img-1):
        for j in range(w_img-1):
            # Extract 2x2 region
            region = gray_img[i:i+2, j:j+2]
            # Calculate convolution results
            gx = np.sum(region * roberts_x)
            gy = np.sum(region * roberts_y)
            # Calculate gradient magnitude
            result[i, j] = sqrt(gx**2 + gy**2)
    
    # Normalize to range 0-255
    max_magnitude = np.max(result)
    if max_magnitude > 0:
        result = (result / max_magnitude) * 255
    
    return result.astype(np.uint8)

def canny_edge_detection(img, low_threshold=15, high_threshold=40):
    """
    Detect edges using Canny edge detector
    
    Parameters:
    img (numpy.ndarray): Input image
    low_threshold (int): Lower threshold for edge detection
    high_threshold (int): Higher threshold for edge detection
    
    Returns:
    numpy.ndarray: Edge map
    """
    # Ensure grayscale image
    if len(img.shape) == 3:
        gray_img = grayscale(img)
    else:
        gray_img = img.copy()
    
    # Step 1: Noise reduction with Gaussian filter
    gauss = (1.0/57) * np.array([
        [0, 1, 2, 1, 0],
        [1, 3, 5, 3, 1],
        [2, 5, 9, 5, 2],
        [1, 3, 5, 3, 1],
        [0, 1, 2, 1, 0]], dtype=np.float32)
    
    smoothed = convolution(gray_img, gauss)
    
    # Step 2: Gradient calculation using Sobel
    sX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], dtype=np.float32)
    sY = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]], dtype=np.float32)
    
    # gx = convolution(smoothed, sX)
    # gy = convolution(smoothed, sY)

    gx = cv2.filter2D(smoothed, cv2.CV_64F, sX)
    gy = cv2.filter2D(smoothed, cv2.CV_64F, sY)
    
    # Calculate gradient magnitude and direction
    grad = np.sqrt(gx**2 + gy**2)
    grad = (grad / grad.max()) * 255
    theta = np.arctan2(gy, gx)
    
    # Step 3: Non-maximum suppression
    H, W = grad.shape
    nonMax = np.zeros((H, W), dtype=np.uint8)
    
    # Convert angle to degrees and handle negative angles
    angle = theta * 180 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, H-1):
        for j in range(1, W-1):
            q = 255
            r = 255
            
            # Classify gradient direction
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = grad[i, j+1]
                r = grad[i, j-1]
            elif (22.5 <= angle[i, j] < 67.5):
                q = grad[i+1, j-1]
                r = grad[i-1, j+1]
            elif (67.5 <= angle[i, j] < 112.5):
                q = grad[i+1, j]
                r = grad[i-1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q = grad[i-1, j-1]
                r = grad[i+1, j+1]
            
            # Remove non-maximum pixels
            if (grad[i,j] >= q) and (grad[i,j] >= r):
                nonMax[i,j] = grad[i,j]
            else:
                nonMax[i,j] = 0
    
    # Step 4: Hysteresis thresholding
    result = np.zeros((H, W), dtype=np.uint8)
    
    # Initial classification of edges based on thresholds
    for i in range(H):
        for j in range(W):
            pixel = nonMax[i, j]
            if pixel >= high_threshold:  # Strong edge
                result[i, j] = 255
            elif pixel >= low_threshold:  # Weak edge
                result[i, j] = low_threshold
            else:  # Not an edge
                result[i, j] = 0
    
    # Step 5: Edge tracking by hysteresis
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if result[i, j] == low_threshold:
                try:
                    # Check if weak edge is connected to a strong edge
                    if ((result[i + 1, j - 1] == 255) or (result[i + 1, j] == 255) or (result[i + 1, j + 1] == 255) or
                        (result[i, j - 1] == 255) or (result[i, j + 1] == 255) or
                        (result[i - 1, j - 1] == 255) or (result[i - 1, j] == 255) or (result[i - 1, j + 1] == 255)):
                        result[i, j] = 255  # Keep as strong edge
                    else:
                        result[i, j] = 0  # Remove weak edge
                except IndexError:
                    pass
    
    return result

def dft_edge_detection(img):
    """
    Detect edges using DFT-based high-pass filter
    
    Parameters:
    img (numpy.ndarray): Input image
    
    Returns:
    numpy.ndarray: Edge map
    """
    # Ensure grayscale image
    if len(img.shape) == 3:
        gray_img = grayscale(img)
    else:
        gray_img = img.copy()
    
    # Perform DFT
    dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Create high-pass filter mask
    rows, cols = gray_img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 80  # Radius of filter
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0
    
    # Apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Normalize result
    img_back_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    
    return img_back_normalized.astype(np.uint8)
