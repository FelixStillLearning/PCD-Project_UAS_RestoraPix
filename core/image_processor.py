"""
Core Image Processor module
Contains base functionality for image processing operations
"""
import cv2
import numpy as np

class ImageProcessor:
    """
    Core image processing class that provides basic functionality
    for loading, saving, and displaying images.
    """
    def __init__(self):
        """Initialize the image processor"""
        self.image = None
        self.original_image = None
        self.processed_image = None
        self.undo_stack = []
        self.redo_stack = []
        self.max_stack_size = 10  # Maximum number of images to store in undo/redo stack

    def load_image(self, file_path):
        """
        Load an image from file
        
        Parameters:
        file_path (str): Path to the image file
        
        Returns:
        numpy.ndarray: Loaded image
        """
        self.image = cv2.imread(file_path)
        if self.image is not None:
            self.original_image = self.image.copy()
        return self.image

    def save_image(self, file_path):
        """
        Save the current image to file
        
        Parameters:
        file_path (str): Output file path
        
        Returns:
        bool: True if successful, False otherwise
        """
        if self.image is None:
            return False
            
        # Add file extension if not provided
        if not any(file_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']):
            file_path += '.jpg'
                
        # Save the image
        try:
            cv2.imwrite(file_path, self.image)
            return True
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return False

    def reset(self):
        """
        Reset to original image
        
        Returns:
        numpy.ndarray: Original image
        """
        if self.original_image is not None:
            self.image = self.original_image.copy()
        return self.image

    def get_image(self):
        """
        Get the current image
        
        Returns:
        numpy.ndarray: Current image
        """
        return self.image    

    def set_image(self, image):
        """
        Set the current image
        
        Parameters:
        image (numpy.ndarray): Image to set as current
        """
        if self.image is not None:
            # Store current image for undo
            self.undo_stack.append(self.image.copy())
            if len(self.undo_stack) > self.max_stack_size:
                self.undo_stack.pop(0)  # Remove oldest image if stack is full
            
            # Clear redo stack when setting a new image
            self.redo_stack.clear()
            
        self.image = image
        
    def undo(self):
        """
        Undo the last image processing operation
        
        Returns:
        numpy.ndarray: Previous image, or None if undo is not possible
        """
        if not self.undo_stack:
            return None
            
        # Store current image for redo
        self.redo_stack.append(self.image.copy())
        
        # Pop last image from undo stack
        previous_image = self.undo_stack.pop()
        self.image = previous_image
        
        return previous_image
        
    def redo(self):
        """
        Redo the last undone operation
        
        Returns:
        numpy.ndarray: Next image, or None if redo is not possible
        """
        if not self.redo_stack:
            return None
            
        # Store current image for undo
        self.undo_stack.append(self.image.copy())
        
        # Pop from redo stack
        next_image = self.redo_stack.pop()
        self.image = next_image
        
        return next_image
        
    def to_grayscale(self, image=None):
        """
        Convert image to grayscale
        
        Parameters:
        image (numpy.ndarray, optional): Input image. If None, use self.image
        
        Returns:
        numpy.ndarray: Grayscale image
        """
        if image is None:
            image = self.image
            
        if image is None:
            return None
            
        # Return image if already grayscale
        if len(image.shape) == 2:
            return image
            
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
