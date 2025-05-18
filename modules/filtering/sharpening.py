"""
Sharpening module
Contains functions for image sharpening operations
"""
import numpy as np
import cv2
from ..filtering.filters import convolution

def sharpen(img, kernel_type="basic"):
    """
    Sharpen an image using different kernel types
    
    Parameters:
    img (numpy.ndarray): Input image
    kernel_type (str): Type of sharpening kernel to use
                      Options: "basic", "strong", "mild", "laplacian", "high_boost", "custom"
    
    Returns:
    numpy.ndarray: Sharpened image
    """
    # Ensure grayscale image
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()

    # Define different sharpening kernels
    kernels = {
        "basic": np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ], np.float32),
        
        "strong": np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], np.float32),
        
        "mild": np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], np.float32),
        
        "laplacian": np.array([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ], np.float32),
        
        "high_boost": np.array([
            [1, -2, 1],
            [-2, 5, -2],
            [1, -2, 1]
        ], np.float32),
        
        "custom": (1/16) * np.array([
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]
        ], np.float32)
    }
    
    # Get the specified kernel
    if kernel_type not in kernels:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Available types: {list(kernels.keys())}")
    
    kernel = kernels[kernel_type]
    
    # Apply convolution with sharpening kernel
    return convolution(gray_img, kernel)

def unsharp_masking(img, amount=1.5, kernel_size=5, sigma=1.0):
    """
    Apply unsharp masking to an image
    
    Parameters:
    img (numpy.ndarray): Input image
    amount (float): Amount of sharpening (1.0 to 3.0)
    kernel_size (int): Size of Gaussian blur kernel
    sigma (float): Sigma value for Gaussian blur
    
    Returns:
    numpy.ndarray: Sharpened image
    """
    # Ensure grayscale image
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()
    
    # Create blurred version of image
    blurred = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), sigma)
    
    # Calculate unsharp mask
    unsharp_mask = cv2.addWeighted(gray_img, 1 + amount, blurred, -amount, 0)
    
    return unsharp_mask

def high_pass_filter(img):
    """
    Apply high-pass filter to an image using frequency domain filtering
    
    Parameters:
    img (numpy.ndarray): Input image
    
    Returns:
    numpy.ndarray: High-pass filtered image
    """
    # Ensure grayscale image
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()
    
    # Get image dimensions
    rows, cols = gray_img.shape
    
    # Create a high-pass filter mask
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.float32)
    r = 30  # Radius of the filter
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0
    
    # Perform DFT on the image
    dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Apply high-pass filter in frequency domain
    fshift = dft_shift * mask
    
    # Perform inverse DFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Normalize to 0-255 range
    img_back_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    
    return img_back_normalized.astype(np.uint8)
