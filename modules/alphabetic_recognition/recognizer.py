"""
Alphabetic Recognition Module
============================

Module untuk integrasi alphabetic recognition dengan aplikasi utama.
Menggunakan model yang sudah di-train untuk klasifikasi karakter A-Z, 0-9.
"""

import cv2
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Optional, List
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

class AlphabeticRecognizer:
    """
    Class untuk alphabetic recognition menggunakan model yang sudah di-train.
    """
    
    def __init__(self, model_path: str = "models/alphabetic_classifier_model.pkl"):
        """
        Initialize alphabetic recognizer.
        
        Args:
            model_path: Path ke file model yang sudah di-train
        """
        self.model_path = model_path
        self.model_data = None
        self.classifier = None
        self.classes = None
        self.feature_size = None
        self.is_loaded = False
        
        # Load model
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load model yang sudah di-train.
        
        Returns:
            True jika berhasil load, False jika gagal
        """
        try:
            if not Path(self.model_path).exists():
                print(f"❌ Model file not found: {self.model_path}")
                return False
            
            # Load model data
            self.model_data = joblib.load(self.model_path)
            self.classifier = self.model_data['model']
            self.classes = self.model_data['classes']
            self.feature_size = self.model_data['feature_size']
            
            self.is_loaded = True
            print(f"✓ Alphabetic recognition model loaded successfully")
            print(f"  Model type: {self.model_data['model_type']}")
            print(f"  Classes: {self.classes}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.is_loaded = False
            return False
    
    def preprocess_character_image(self, image_roi: np.ndarray, target_size: Tuple[int, int] = (28, 28)) -> np.ndarray:
        """
        Preprocess karakter ROI untuk feature extraction.
        Sama seperti preprocessing dalam training script.
        
        Args:
            image_roi: ROI image yang berisi karakter
            target_size: Target size untuk normalisasi
            
        Returns:
            Preprocessed binary image
        """
        try:
            # Convert to grayscale if needed
            if len(image_roi.shape) == 3:
                gray_image = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image_roi.copy()
            
            # Noise reduction
            denoised = cv2.GaussianBlur(gray_image, (3, 3), 1.0)
            denoised = cv2.medianBlur(denoised, 3)
            
            # Binarization using Otsu
            _, binary_image = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            # Resize to target size
            normalized = cv2.resize(cleaned, target_size, interpolation=cv2.INTER_AREA)
            
            # Ensure proper orientation (background black, foreground white)
            white_pixels = np.count_nonzero(normalized == 255)
            total_pixels = normalized.shape[0] * normalized.shape[1]
            
            if white_pixels > total_pixels * 0.5:
                normalized = cv2.bitwise_not(normalized)
            
            return normalized
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            # Fallback preprocessing
            if len(image_roi.shape) == 3:
                gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_roi.copy()
            
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            resized = cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)
            return resized
    
    def extract_features_from_character(self, binary_char_image: np.ndarray) -> np.ndarray:
        """
        Extract features dari preprocessed character image.
        Sama seperti feature extraction dalam training script.
        
        Args:
            binary_char_image: Binary character image (28x28)
            
        Returns:
            Feature vector
        """
        features = []
        
        try:
            # 1. Hu Moments
            moments = cv2.moments(binary_char_image)
            hu_moments = cv2.HuMoments(moments)
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            features.extend(hu_moments.flatten())
            
            # 2. HOG Features
            hog = cv2.HOGDescriptor(
                _winSize=(28, 28),
                _blockSize=(14, 14),
                _blockStride=(7, 7),
                _cellSize=(7, 7),
                _nbins=9
            )
            hog_features = hog.compute(binary_char_image)
            features.extend(hog_features.flatten())
            
            # 3. Geometric Features
            contours, _ = cv2.findContours(binary_char_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                features.append(aspect_ratio)
                
                char_area = cv2.contourArea(largest_contour)
                bbox_area = w * h
                area_ratio = char_area / bbox_area if bbox_area > 0 else 0
                features.append(area_ratio)
                
                perimeter = cv2.arcLength(largest_contour, True)
                perimeter_area_ratio = perimeter / char_area if char_area > 0 else 0
                features.append(perimeter_area_ratio)
                
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                solidity = char_area / hull_area if hull_area > 0 else 0
                features.append(solidity)
                
                extent = char_area / bbox_area if bbox_area > 0 else 0
                features.append(extent)
            else:
                features.extend([0, 0, 0, 0, 0])
            
            # 4. Projection Features
            h_projection = np.sum(binary_char_image, axis=1)
            v_projection = np.sum(binary_char_image, axis=0)
            
            features.extend([
                np.mean(h_projection), np.std(h_projection), np.max(h_projection),
                np.mean(v_projection), np.std(v_projection), np.max(v_projection)
            ])
            
            # 5. Zoning Features
            zone_h, zone_w = binary_char_image.shape[0] // 4, binary_char_image.shape[1] // 4
            for i in range(4):
                for j in range(4):
                    zone = binary_char_image[i*zone_h:(i+1)*zone_h, j*zone_w:(j+1)*zone_w]
                    zone_density = np.sum(zone) / (zone_h * zone_w * 255) if zone.size > 0 else 0
                    features.append(zone_density)
            
            # 6. Crossing Features
            h_crossings = 0
            for row in binary_char_image:
                crossings = 0
                for i in range(len(row) - 1):
                    if row[i] != row[i + 1]:
                        crossings += 1
                h_crossings += crossings
            
            v_crossings = 0
            for col in range(binary_char_image.shape[1]):
                column = binary_char_image[:, col]
                crossings = 0
                for i in range(len(column) - 1):
                    if column[i] != column[i + 1]:
                        crossings += 1
                v_crossings += crossings
            
            features.extend([h_crossings, v_crossings])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            # Return default feature vector
            return np.zeros(self.feature_size if self.feature_size else 100, dtype=np.float32)
    
    def predict_character(self, image_roi: np.ndarray) -> Tuple[str, float]:
        """
        Predict karakter dari ROI image.
        
        Args:
            image_roi: ROI image yang berisi karakter
            
        Returns:
            Tuple (predicted_character, confidence_score)
        """
        if not self.is_loaded:
            return "?", 0.0
        
        try:
            # Preprocess image
            preprocessed = self.preprocess_character_image(image_roi)
            
            # Extract features
            features = self.extract_features_from_character(preprocessed)
            
            # Make prediction
            prediction = self.classifier.predict([features])[0]
            
            # Get confidence if available
            confidence = 1.0
            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba([features])[0]
                confidence = np.max(probabilities)
            elif hasattr(self.classifier, 'decision_function'):
                # For SVM
                decision_scores = self.classifier.decision_function([features])
                if len(self.classes) == 2:
                    confidence = abs(decision_scores[0])
                else:
                    confidence = np.max(decision_scores)
                # Normalize confidence score
                confidence = min(1.0, confidence / 2.0)
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Error in character prediction: {e}")
            return "?", 0.0
    
    def predict_characters_from_image(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[str, float]]:
        """
        Predict multiple characters dari bounding boxes dalam satu image.
        
        Args:
            image: Original image
            bboxes: List of bounding boxes [(x, y, w, h), ...]
            
        Returns:
            List of (character, confidence) tuples
        """
        results = []
        
        for bbox in bboxes:
            x, y, w, h = bbox
            
            # Extract ROI
            roi = image[y:y+h, x:x+w]
            
            # Predict character
            char, conf = self.predict_character(roi)
            results.append((char, conf))
        
        return results
    
    def get_model_info(self) -> dict:
        """
        Get informasi tentang model yang di-load.
        
        Returns:
            Dictionary berisi informasi model
        """
        if not self.is_loaded:
            return {}
        
        return {
            'model_type': self.model_data.get('model_type', 'Unknown'),
            'accuracy': self.model_data.get('accuracy', 0.0),
            'classes': self.classes,
            'feature_size': self.feature_size,
            'total_classes': len(self.classes) if self.classes else 0
        }

# Global instance untuk digunakan dalam aplikasi
_recognizer_instance = None

def get_alphabetic_recognizer() -> AlphabeticRecognizer:
    """
    Get singleton instance dari AlphabeticRecognizer.
    
    Returns:
        AlphabeticRecognizer instance
    """
    global _recognizer_instance
    if _recognizer_instance is None:
        _recognizer_instance = AlphabeticRecognizer()
    return _recognizer_instance

def predict_character(image_roi: np.ndarray) -> Tuple[str, float]:
    """
    Helper function untuk predict single character.
    
    Args:
        image_roi: ROI image yang berisi karakter
        
    Returns:
        Tuple (character, confidence)
    """
    recognizer = get_alphabetic_recognizer()
    return recognizer.predict_character(image_roi)

def predict_characters(image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[str, float]]:
    """
    Helper function untuk predict multiple characters.
    
    Args:
        image: Original image
        bboxes: List of bounding boxes
        
    Returns:
        List of (character, confidence) tuples
    """
    recognizer = get_alphabetic_recognizer()
    return recognizer.predict_characters_from_image(image, bboxes)

# Test function
if __name__ == "__main__":
    print("Testing Alphabetic Recognition Module...")
    
    # Test dengan dummy image
    dummy_image = np.ones((50, 50, 3), dtype=np.uint8) * 255
    cv2.rectangle(dummy_image, (10, 10), (40, 40), (0, 0, 0), 2)
    
    char, conf = predict_character(dummy_image)
    print(f"Prediction: {char}, Confidence: {conf:.4f}")
    
    # Get model info
    recognizer = get_alphabetic_recognizer()
    info = recognizer.get_model_info()
    print(f"Model info: {info}")
