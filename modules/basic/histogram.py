"""
Histogram module
Contains functions for histogram generation and manipulation
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

from modules.basic.operations import grayscale

def calculate_grayscale_histogram(image):
    """
    Calculate histogram of a grayscale image
    
    Parameters:
    image (numpy.ndarray): Input grayscale image
    
    Returns:
    numpy.ndarray: Histogram values
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = grayscale(image)
    else:
        gray_image = image.copy()
    
    # Calculate histogram using numpy
    hist = np.zeros(256)
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            hist[gray_image[i, j]] += 1
            
    return hist

def calculate_rgb_histogram(image):
    """
    Calculate histogram for each channel of RGB image
    
    Parameters:
    image (numpy.ndarray): Input RGB image
    
    Returns:
    tuple: (blue_hist, green_hist, red_hist) - Histograms for each channel
    """
    if len(image.shape) != 3:
        raise ValueError("Input image must be RGB (3 channels)")
    
    # Calculate histogram for each channel
    blue_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    green_hist = cv2.calcHist([image], [1], None, [256], [0, 256])
    red_hist = cv2.calcHist([image], [2], None, [256], [0, 256])
    
    return blue_hist, green_hist, red_hist

def calculate_equalized_histogram(image):
    """
    Calculate equalized image and its histogram/CDF for a grayscale image.
    Parameters:
        image (numpy.ndarray): Input grayscale image
    Returns:
        tuple: (equalized_image, cdf_normalized, hist)
    """
    # Pastikan image grayscale
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_uint8 = np.ma.filled(cdf_m, 0).astype('uint8')
    equalized_image = cdf_uint8[image]
    return equalized_image, cdf_normalized, hist

def plot_grayscale_histogram(image, title="Grayscale Histogram"):
    """
    Plot histogram of a grayscale image
    
    Parameters:
    image (numpy.ndarray): Input grayscale image
    title (str): Title for the plot
    
    Returns:
    matplotlib.figure.Figure: The figure object
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = grayscale(image)
    else:
        gray_image = image.copy()
    
    # Create figure
    fig = plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    # Plot histogram
    plt.hist(gray_image.ravel(), 256, [0, 256])
    
    return fig

def plot_rgb_histogram(image, title="RGB Histogram"):
    """
    Plot histogram for each channel of RGB image
    
    Parameters:
    image (numpy.ndarray): Input RGB image
    title (str): Title for the plot
    
    Returns:
    matplotlib.figure.Figure: The figure object
    """
    if len(image.shape) != 3:
        raise ValueError("Input image must be RGB (3 channels)")
    
    # Create figure
    fig = plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    # Calculate and plot histogram for each channel
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    
    plt.xlim([0, 256])
    plt.legend(('Blue', 'Green', 'Red'))
    
    return fig

def plot_equalized_histogram(image):
    """
    Plot histogram and CDF of an image (after equalization)
    """
    # Hitung hasil equalization dan data histogram/CDF
    equalized_image, cdf_normalized, hist = calculate_equalized_histogram(image)
    plt.figure()
    plt.plot(cdf_normalized, color='b')
    plt.hist(equalized_image.flatten(), 256, [0, 256], color='r', alpha=0.5)
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()
    return None


