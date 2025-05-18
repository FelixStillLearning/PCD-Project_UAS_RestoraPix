"""
Frequency Domain module
Contains functions for frequency domain image processing operations
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def dft(image):
    """
    Compute the Discrete Fourier Transform of an image
    
    Parameters:
    image (numpy.ndarray): Input image
    
    Returns:
    tuple: (dft_shift, magnitude_spectrum)
    """
    # Ensure grayscale image
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image.copy()
    
    # Perform DFT
    dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Calculate magnitude spectrum
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-5)
    
    return dft_shift, magnitude_spectrum

def low_pass_filter(image, radius=50):
    """
    Apply low-pass filter to an image in frequency domain
    
    Parameters:
    image (numpy.ndarray): Input image
    radius (int): Radius of the low-pass filter
    
    Returns:
    numpy.ndarray: Filtered image
    tuple: (filtered_image, magnitude_spectrum, filtered_magnitude_spectrum)
    """
    # Ensure grayscale image
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image.copy()
    
    # Perform DFT
    dft_shift, magnitude_spectrum = dft(gray_img)
    
    # Create mask for low-pass filter
    rows, cols = gray_img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius * radius
    mask[mask_area] = 1
    
    # Apply mask and inverse DFT
    fshift = dft_shift * mask
    fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 1e-5)
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Normalize result
    img_back_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    
    return img_back_normalized.astype(np.uint8), magnitude_spectrum, fshift_mask_mag

def high_pass_filter(image, radius=50):
    """
    Apply high-pass filter to an image in frequency domain
    
    Parameters:
    image (numpy.ndarray): Input image
    radius (int): Radius of the high-pass filter
    
    Returns:
    numpy.ndarray: Filtered image
    tuple: (filtered_image, magnitude_spectrum, filtered_magnitude_spectrum)
    """
    # Ensure grayscale image
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image.copy()
    
    # Perform DFT
    dft_shift, magnitude_spectrum = dft(gray_img)
    
    # Create mask for high-pass filter
    rows, cols = gray_img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius * radius
    mask[mask_area] = 0
    
    # Apply mask and inverse DFT
    fshift = dft_shift * mask
    fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 1e-5)
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Normalize result
    img_back_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    
    return img_back_normalized.astype(np.uint8), magnitude_spectrum, fshift_mask_mag

def band_pass_filter(image, inner_radius=30, outer_radius=80):
    """
    Apply band-pass filter to an image in frequency domain
    
    Parameters:
    image (numpy.ndarray): Input image
    inner_radius (int): Inner radius of the band-pass filter
    outer_radius (int): Outer radius of the band-pass filter
    
    Returns:
    numpy.ndarray: Filtered image
    tuple: (filtered_image, magnitude_spectrum, filtered_magnitude_spectrum)
    """
    # Ensure grayscale image
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image.copy()
    
    # Perform DFT
    dft_shift, magnitude_spectrum = dft(gray_img)
    
    # Create mask for band-pass filter
    rows, cols = gray_img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    
    # Define the band region
    mask_area = np.logical_and(
        (x - center[0]) ** 2 + (y - center[1]) ** 2 >= inner_radius ** 2,
        (x - center[0]) ** 2 + (y - center[1]) ** 2 <= outer_radius ** 2
    )
    mask[mask_area] = 1
    
    # Apply mask and inverse DFT
    fshift = dft_shift * mask
    fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 1e-5)
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Normalize result
    img_back_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    
    return img_back_normalized.astype(np.uint8), magnitude_spectrum, fshift_mask_mag

def plot_frequency_domain_result(original, magnitude_spectrum, filtered_spectrum, result):
    """
    Plot the results of frequency domain filtering
    
    Parameters:
    original (numpy.ndarray): Original image
    magnitude_spectrum (numpy.ndarray): Magnitude spectrum of original image
    filtered_spectrum (numpy.ndarray): Magnitude spectrum after filtering
    result (numpy.ndarray): Result image after filtering
    
    Returns:
    matplotlib.figure.Figure: Figure with plots
    """
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(original, cmap='gray')
    ax1.title.set_text('Input Image')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(magnitude_spectrum, cmap='gray')
    ax2.title.set_text('FFT of Image')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(filtered_spectrum, cmap='gray')
    ax3.title.set_text('FFT + Mask')
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(result, cmap='gray')
    ax4.title.set_text('Filtered Image')
    
    return fig