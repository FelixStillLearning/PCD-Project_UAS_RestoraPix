"""
Core Image Processor module
Contains base functionality for image processing operations
"""
import cv2
import numpy as np

# Import alphabetic recognition module
try:
    from modules.alphabetic_recognition.recognizer import AlphabeticRecognizer, predict_character
    ALPHABETIC_RECOGNITION_AVAILABLE = True
except ImportError:
    ALPHABETIC_RECOGNITION_AVAILABLE = False
    print("Warning: Alphabetic recognition module not available")

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
        
        # Initialize alphabetic recognizer
        self.alphabetic_recognizer = None
        if ALPHABETIC_RECOGNITION_AVAILABLE:
            try:
                self.alphabetic_recognizer = AlphabeticRecognizer()
            except Exception as e:
                print(f"Warning: Could not initialize alphabetic recognizer: {e}")

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

    def export_pixel_data(self, file_path, format_type):
        """
        Export pixel data of current image to a file
        
        Parameters:
        file_path (str): Path to save the file
        format_type (str): Format type ('txt', 'csv', or 'xlsx')
        
        Returns:
        bool: True if successful, False otherwise
        """
        if self.image is None:
            return False
            
        try:
            # Get image dimensions and data
            if len(self.image.shape) == 2:  # Grayscale
                height, width = self.image.shape
                pixel_data = self.image.reshape(height * width, 1)
                columns = ["Gray"]
            else:  # RGB
                height, width, _ = self.image.shape
                pixel_data = self.image.reshape(height * width, 3)
                columns = ["Red", "Green", "Blue"]
                
            # Export based on format
            if format_type == 'txt':
                np.savetxt(file_path, pixel_data, fmt="%d", delimiter=",")
            elif format_type == 'csv':
                import pandas as pd
                pd.DataFrame(pixel_data, columns=columns).to_csv(file_path, index=False)
            elif format_type == 'xlsx':
                import pandas as pd
                pd.DataFrame(pixel_data, columns=columns).to_excel(file_path, index=False, engine="openpyxl")
            else:
                return False
                
            return True
        except Exception as e:
            print(f"Error exporting pixel data: {str(e)}")
            return False

    def process_for_alphabetic_recognition(self, image=None, min_contour_area=100):
        """
        Process image for alphabetic recognition by detecting character regions
        and applying character recognition to each region.
        
        Parameters:
        image (numpy.ndarray): Image to process (if None, uses current image)
        min_contour_area (int): Minimum contour area to consider as character
        
        Returns:
        tuple: (processed_image_with_annotations, detection_results)
               detection_results is list of (character, confidence, bbox) tuples
        """
        if image is None:
            image = self.image
            
        if image is None:
            return None, []
            
        if not ALPHABETIC_RECOGNITION_AVAILABLE or self.alphabetic_recognizer is None:
            print("Alphabetic recognition not available")
            return image.copy(), []
        
        try:
            # Convert to grayscale for contour detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply binary threshold to find characters
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create result image
            result_image = image.copy()
            detection_results = []
            
            # Process each contour
            for contour in contours:
                # Filter small contours
                area = cv2.contourArea(contour)
                if area < min_contour_area:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract character region
                char_roi = image[y:y+h, x:x+w]
                
                if char_roi.size == 0:
                    continue
                
                # Predict character
                try:
                    character, confidence = self.alphabetic_recognizer.predict_character(char_roi)
                    
                    # Only show results with reasonable confidence
                    if confidence > 0.3:
                        # Draw bounding box
                        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Add text label
                        label = f"{character}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(result_image, (x, y-25), (x+label_size[0], y), (0, 255, 0), -1)
                        cv2.putText(result_image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                          # Store result as dictionary for consistency
                        detection_results.append({
                            'character': character,
                            'confidence': confidence,
                            'bbox': (x, y, w, h)
                        })
                
                except Exception as e:
                    print(f"Error predicting character: {e}")
                    continue
            
            return result_image, detection_results
            
        except Exception as e:
            print(f"Error in alphabetic recognition: {e}")
            return image.copy(), []