"""
Utility helpers module
Contains utility functions for image processing operations
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

def load_image(file_path):
    """
    Load an image from file
    
    Parameters:
    file_path (str): Path to the image file
    
    Returns:
    numpy.ndarray: Loaded image or None if failed
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist")
        return None
    
    image = cv2.imread(file_path)
    if image is None:
        print(f"Error: Could not read image from '{file_path}'")
    
    return image

def save_image(image, file_path, create_dirs=True):
    """
    Save an image to file
    
    Parameters:
    image (numpy.ndarray): Image to save
    file_path (str): Output file path
    create_dirs (bool): Create directories if they don't exist
    
    Returns:
    bool: True if successful, False otherwise
    """
    if image is None:
        print("Error: Cannot save None image")
        return False
    
    # Create directories if they don't exist
    if create_dirs:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                print(f"Error creating directories: {str(e)}")
                return False
    
    # Add file extension if not provided
    if not any(file_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']):
        file_path += '.jpg'
    
    try:
        cv2.imwrite(file_path, image)
        return True
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return False

def to_grayscale(image):
    """
    Convert an RGB image to grayscale
    
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
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def display_image(image, title="Image", wait_key=0):
    """
    Display an image using OpenCV
    
    Parameters:
    image (numpy.ndarray): Image to display
    title (str): Window title
    wait_key (int): Time in milliseconds to wait (0 = wait indefinitely)
    """
    if image is None:
        print("Error: Cannot display None image")
        return
    
    cv2.imshow(title, image)
    cv2.waitKey(wait_key)

def display_multiple_images(images, titles=None, rows=1, cols=None, figsize=(15, 10)):
    """
    Display multiple images using matplotlib
    
    Parameters:
    images (list): List of images to display
    titles (list): List of titles for each image (optional)
    rows (int): Number of rows in the subplot grid
    cols (int): Number of columns in the subplot grid (calculated if None)
    figsize (tuple): Figure size (width, height) in inches
    
    Returns:
    matplotlib.figure.Figure: Figure with subplots
    """
    if not images:
        print("Error: No images to display")
        return None
    
    # Calculate number of columns if not provided
    if cols is None:
        cols = (len(images) + rows - 1) // rows
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Display each image
    for i, image in enumerate(images):
        if i >= rows * cols:
            break
        
        ax = fig.add_subplot(rows, cols, i + 1)
        
        # Set title if available
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])
        
        # Display image (convert BGR to RGB if color image)
        if image is not None:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ax.imshow(image_rgb)
            else:
                ax.imshow(image, cmap='gray')
        
        # Turn off axis
        ax.axis('off')
    
    # Adjust layout and show
    plt.tight_layout()
    
    return fig

def compare_images(original, processed, title1="Original", title2="Processed"):
    """
    Display original and processed images side by side
    
    Parameters:
    original (numpy.ndarray): Original image
    processed (numpy.ndarray): Processed image
    title1 (str): Title for the original image
    title2 (str): Title for the processed image
    
    Returns:
    matplotlib.figure.Figure: Figure with comparison
    """
    return display_multiple_images([original, processed], [title1, title2], rows=1, cols=2)

def generate_timestamp():
    """
    Generate a timestamp string for file naming
    
    Returns:
    str: Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_kernel(kernel_type, size=3):
    """
    Get a predefined kernel for image processing
    
    Parameters:
    kernel_type (str): Type of kernel ('mean', 'gaussian', 'laplacian', 'sobel_x', 'sobel_y', 'prewitt_x', 'prewitt_y')
    size (int): Size of the kernel (3, 5, etc.)
    
    Returns:
    numpy.ndarray: Kernel matrix
    """
    if kernel_type == 'mean':
        return np.ones((size, size), np.float32) / (size * size)
    
    elif kernel_type == 'gaussian':
        if size == 3:
            return np.array([
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]
            ], np.float32) / 16
        elif size == 5:
            return np.array([
                [1, 4, 7, 4, 1],
                [4, 16, 26, 16, 4],
                [7, 26, 41, 26, 7],
                [4, 16, 26, 16, 4],
                [1, 4, 7, 4, 1]
            ], np.float32) / 273
        else:
            raise ValueError(f"Unsupported size {size} for gaussian kernel. Use 3 or 5.")
    
    elif kernel_type == 'laplacian':
        return np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], np.float32)
    
    elif kernel_type == 'sobel_x':
        return np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], np.float32)
    
    elif kernel_type == 'sobel_y':
        return np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], np.float32)
    
    elif kernel_type == 'prewitt_x':
        return np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], np.float32)
    
    elif kernel_type == 'prewitt_y':
        return np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ], np.float32)
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

def compute_histogram(image, channels=None, mask=None):
    """
    Compute histogram for an image
    
    Parameters:
    image (numpy.ndarray): Input image
    channels (list): List of channels to compute histograms for
    mask (numpy.ndarray): Mask for histogram calculation
    
    Returns:
    list: List of histograms for each channel
    """
    if image is None:
        return None
    
    # Set default channels if not provided
    if channels is None:
        if len(image.shape) == 2:  # Grayscale image
            channels = [0]
        else:  # Color image
            channels = [0, 1, 2]
    
    # Compute histograms
    histograms = []
    for channel in channels:
        hist = cv2.calcHist([image], [channel], mask, [256], [0, 256])
        histograms.append(hist)
    
    return histograms