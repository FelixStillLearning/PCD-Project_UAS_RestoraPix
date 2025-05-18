"""
Restoration operations module
Contains functions for image restoration like inpainting, deblurring and old photo restoration
"""
import cv2
import numpy as np
from scipy import ndimage

def inpainting(image, mask=None):
    """
    Remove scratches or defects from an image using inpainting
    
    Parameters:
    image (numpy.ndarray): Input image
    mask (numpy.ndarray): Optional mask image (if None, a simple threshold-based mask will be created)
    
    Returns:
    numpy.ndarray: Inpainted image
    """
    # If no mask is provided, create a simple one based on thresholding
    if mask is None:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Create a simple mask based on thresholding
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # Dilate the mask to ensure coverage of defects
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
    # Apply inpainting
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    
    return result

def deblurring(image, kernel_size=5, sigma=0):
    """
    Sharpen a blurry image using deconvolution techniques
    
    Parameters:
    image (numpy.ndarray): Blurry input image
    kernel_size (int): Size of the deconvolution kernel
    sigma (float): Sigma value for the unsharp mask
    
    Returns:
    numpy.ndarray: Deblurred image
    """
    # Convert to grayscale if color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Apply Wiener deconvolution (simplified version)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    deblurred = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    
    # If original was color, convert back to color
    if len(image.shape) == 3:
        # Apply the same enhancement to each channel
        result = image.copy()
        for i in range(3):
            blurred_channel = cv2.GaussianBlur(image[:,:,i], (kernel_size, kernel_size), sigma)
            result[:,:,i] = cv2.addWeighted(image[:,:,i], 1.5, blurred_channel, -0.5, 0)
        return result
    
    return deblurred

def old_photo_restoration(image):
    """
    Restore an old or damaged photo by removing scratches, enhancing contrast, and reducing noise
    
    Parameters:
    image (numpy.ndarray): Old photo image
    
    Returns:
    numpy.ndarray: Restored image
    """
    # Step 1: Apply bilateral filter to smooth while preserving edges
    filtered = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Step 2: Create a mask for scratches/damage (very basic method)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Detect potential scratches using thresholding
    _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    
    # Step 3: Apply inpainting to remove scratches
    restored = cv2.inpaint(filtered, mask, 3, cv2.INPAINT_TELEA)
    
    # Step 4: Enhance contrast
    lab = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge back and convert to BGR
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    return enhanced
