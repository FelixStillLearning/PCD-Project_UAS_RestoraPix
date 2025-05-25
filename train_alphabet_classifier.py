#!/usr/bin/env python3
"""
Enhanced Alphabetic Recognition Training Script
==============================================

Training script for alphabetic character classification using centralized configuration
and optimized parameters for better performance and maintainability.

Author: Enhanced for Project UAS
Date: May 2025
"""

import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import glob
from typing import Tuple, List, Dict, Union, Optional
import warnings
from pathlib import Path

# Import centralized configuration
from modules.alphabetic_recognition.config import (
    MODEL_PATH, FEATURE_CONFIG_PATH, DATASET_PATH, CHAR_IMAGE_SIZE,
    HOG_PARAMS, FEATURE_WEIGHTS, SVM_C_PARAM, SVM_GAMMA,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT, RF_MIN_SAMPLES_LEAF,
    CV_FOLDS, RANDOM_STATE, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA_X,
    MEDIAN_FILTER_SIZE, THRESHOLD_TYPE, THRESHOLD_MAX_VALUE,
    MORPH_OPEN_KERNEL, MORPH_CLOSE_KERNEL, MORPH_ITERATIONS,
    get_preprocessing_config, get_feature_extraction_config,
    get_classification_config, validate_config
)

# Import existing PCD modules with fallbacks
try:
    from modules.color_processing.color_processor import to_grayscale
    from modules.filtering.filters import gaussian_filter, median_filter
    from modules.morphology.operations import opening, closing, erosion, dilation
    from modules.segmentation.thresholding import otsu_threshold, adaptive_threshold
    from modules.transformation.geometric import resize_image
    print("✓ PCD modules imported successfully")
    USE_PCD_MODULES = True
except ImportError as e:
    print(f"⚠ Warning: Could not import some PCD modules: {e}")
    print("  Script will use OpenCV fallbacks where necessary")
    USE_PCD_MODULES = False

# Validate configuration on startup
print("🔧 Validating configuration...")
try:
    validate_config()
    print("✓ Configuration validation passed")
except Exception as e:
    warnings.warn(f"Configuration validation warning: {e}", UserWarning)

# Load configuration dictionaries
PREPROCESSING_CONFIG = get_preprocessing_config()
FEATURE_CONFIG = get_feature_extraction_config()
CLASSIFICATION_CONFIG = get_classification_config()

# Define additional paths for backward compatibility
IMAGES_PATH = str(DATASET_PATH)
ANNOTATIONS_PATH = str(DATASET_PATH / "annotations")

# FNT Dataset path (for reorganized case-safe dataset)
FNT_DATASET_PATH = str(Path("d:/Development/Proyek/Citra/Project_UAS/dataset/alphabets"))

print(f"📋 Training Configuration:")
print(f"  Dataset path: {DATASET_PATH}")
print(f"  Model output: {MODEL_PATH}")
print(f"  Image size: {CHAR_IMAGE_SIZE}")
print(f"  Feature weights: {FEATURE_WEIGHTS}")
print(f"  SVM params: C={SVM_C_PARAM}, gamma={SVM_GAMMA}")
print(f"  RF params: trees={RF_N_ESTIMATORS}, depth={RF_MAX_DEPTH}")

# ==================== ENHANCED PREPROCESSING FUNCTIONS ====================

def preprocess_character_image(image_roi: np.ndarray, 
                              target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Enhanced character preprocessing using centralized configuration.
    
    Parameters:
    -----------
    image_roi : np.ndarray
        ROI from segmented character image
    target_size : Optional[Tuple[int, int]]
        Target size for normalization (uses config default if None)
        
    Returns:
    --------
    np.ndarray
        Clean, normalized binary character image
    """
    if target_size is None:
        target_size = CHAR_IMAGE_SIZE
    
    try:
        # 1. Convert to grayscale using existing modules or fallback
        if len(image_roi.shape) == 3:
            if USE_PCD_MODULES:
                try:
                    gray_image = to_grayscale(image_roi)
                except:
                    gray_image = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image_roi.copy()
        
        # 2. Enhanced noise reduction using configuration
        if USE_PCD_MODULES:
            try:
                denoised = gaussian_filter(gray_image, filter_size=GAUSSIAN_KERNEL_SIZE[0])
            except:
                denoised = cv2.GaussianBlur(gray_image, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA_X)
        else:
            denoised = cv2.GaussianBlur(gray_image, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA_X)
        
        # 3. Median filter for salt-and-pepper noise
        if USE_PCD_MODULES:
            try:
                denoised = median_filter(denoised, filter_size=MEDIAN_FILTER_SIZE)
            except:
                denoised = cv2.medianBlur(denoised, MEDIAN_FILTER_SIZE)
        else:
            denoised = cv2.medianBlur(denoised, MEDIAN_FILTER_SIZE)
        
        # 4. Enhanced binarization using configuration
        if USE_PCD_MODULES:
            try:
                _, binary_image = otsu_threshold(denoised)
            except:
                _, binary_image = cv2.threshold(denoised, 0, THRESHOLD_MAX_VALUE, THRESHOLD_TYPE)
        else:
            _, binary_image = cv2.threshold(denoised, 0, THRESHOLD_MAX_VALUE, THRESHOLD_TYPE)
        
        # 5. Morphological operations using configured kernels
        if USE_PCD_MODULES:
            try:
                # Opening to remove small noise
                cleaned = opening(binary_image, MORPH_OPEN_KERNEL)
                # Closing to fill small holes
                cleaned = closing(cleaned, MORPH_CLOSE_KERNEL)
            except:
                cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, MORPH_OPEN_KERNEL, iterations=MORPH_ITERATIONS)
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, MORPH_CLOSE_KERNEL, iterations=MORPH_ITERATIONS)
        else:
            cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, MORPH_OPEN_KERNEL, iterations=MORPH_ITERATIONS)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, MORPH_CLOSE_KERNEL, iterations=MORPH_ITERATIONS)
          # 6. Resize to target size
        if USE_PCD_MODULES:
            try:
                normalized = resize_image(cleaned, target_size)
            except:
                normalized = cv2.resize(cleaned, target_size, interpolation=cv2.INTER_AREA)
        else:
            normalized = cv2.resize(cleaned, target_size, interpolation=cv2.INTER_AREA)
        
        # 7. Ensure proper orientation (black background, white foreground)
        white_pixels = np.count_nonzero(normalized == 255)
        total_pixels = normalized.shape[0] * normalized.shape[1]
        
        if white_pixels > total_pixels * 0.5:
            normalized = cv2.bitwise_not(normalized)
        
        return normalized
        
    except Exception as e:
        print(f"Error in preprocess_character_image: {e}")
        # Fallback preprocessing using OpenCV only
        if len(image_roi.shape) == 3:
            gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_roi.copy()
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 1.0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        normalized = cv2.resize(cleaned, target_size, interpolation=cv2.INTER_AREA)
        
        return normalized


def extract_features_from_character(binary_char_image: np.ndarray) -> np.ndarray:
    """
    Enhanced feature extraction using centralized configuration and weighted features.
    
    Parameters:
    -----------
    binary_char_image : np.ndarray
        Normalized binary character image
        
    Returns:
    --------
    np.ndarray
        Weighted feature vector
    """
    features = []
    
    try:
        # 1. Enhanced Hu Moments with weighting
        moments = cv2.moments(binary_char_image)
        hu_moments = cv2.HuMoments(moments)
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        weighted_hu = hu_moments.flatten() * FEATURE_WEIGHTS['hu_moments']
        features.extend(weighted_hu)
        
        # 2. Enhanced HOG Features using optimized parameters
        hog = cv2.HOGDescriptor(**HOG_PARAMS)
        hog_features = hog.compute(binary_char_image)
        if hog_features is not None:
            weighted_hog = hog_features.flatten() * FEATURE_WEIGHTS['hog']
            features.extend(weighted_hog)
        else:
            # Fallback HOG with default parameters
            hog_fallback = cv2.HOGDescriptor(
                _winSize=CHAR_IMAGE_SIZE,
                _blockSize=(14, 14),
                _blockStride=(7, 7),
                _cellSize=(7, 7),
                _nbins=9
            )
            hog_features = hog_fallback.compute(binary_char_image)
            weighted_hog = hog_features.flatten() * FEATURE_WEIGHTS['hog']
            features.extend(weighted_hog)
        
        # 3. Enhanced Geometric Features with weighting
        contours, _ = cv2.findContours(binary_char_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Aspect ratio
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Area ratios
            char_area = cv2.contourArea(largest_contour)
            bbox_area = w * h
            area_ratio = char_area / bbox_area if bbox_area > 0 else 0
            
            # Perimeter area ratio
            perimeter = cv2.arcLength(largest_contour, True)
            perimeter_area_ratio = perimeter / char_area if char_area > 0 else 0
            
            # Solidity
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = char_area / hull_area if hull_area > 0 else 0
            
            # Extent
            extent = char_area / bbox_area if bbox_area > 0 else 0
            geometric_features = [aspect_ratio, area_ratio, perimeter_area_ratio, solidity, extent]
        else:
            geometric_features = [0, 0, 0, 0, 0]
        
        weighted_geometric = np.array(geometric_features) * FEATURE_WEIGHTS['geometric']
        features.extend(weighted_geometric)
        
        # 4. Enhanced Projection Features with weighting
        h_projection = np.sum(binary_char_image, axis=1)
        v_projection = np.sum(binary_char_image, axis=0)
        
        projection_features = [
            np.mean(h_projection), np.std(h_projection), np.max(h_projection),
            np.mean(v_projection), np.std(v_projection), np.max(v_projection)
        ]
        weighted_projection = np.array(projection_features) * FEATURE_WEIGHTS['projection']
        features.extend(weighted_projection)
        
        # 5. Enhanced Zoning Features with weighting
        zone_h, zone_w = binary_char_image.shape[0] // 4, binary_char_image.shape[1] // 4
        zoning_features = []
        for i in range(4):
            for j in range(4):
                zone = binary_char_image[i*zone_h:(i+1)*zone_h, j*zone_w:(j+1)*zone_w]
                zone_density = np.sum(zone) / (zone_h * zone_w * 255) if zone.size > 0 else 0
                zoning_features.append(zone_density)
        
        weighted_zoning = np.array(zoning_features) * FEATURE_WEIGHTS['zoning']
        features.extend(weighted_zoning)
        
        # 6. Enhanced Crossing Features with weighting
        h_crossings = 0
        for row in binary_char_image:
            for i in range(len(row) - 1):
                if row[i] != row[i + 1]:
                    h_crossings += 1
        
        v_crossings = 0
        for col in range(binary_char_image.shape[1]):
            column = binary_char_image[:, col]
            for i in range(len(column) - 1):
                if column[i] != column[i + 1]:
                    v_crossings += 1
        
        crossing_features = [h_crossings, v_crossings]
        weighted_crossing = np.array(crossing_features) * FEATURE_WEIGHTS['crossing']
        features.extend(weighted_crossing)
        
        # Validate and clean feature vector
        feature_vector = np.array(features, dtype=np.float32)
        if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
            print("Warning: NaN or Inf values detected in features, applying correction")
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feature_vector
        
    except Exception as e:
        print(f"Error in extract_features_from_character: {e}")
        # Return default feature vector
        return np.zeros(100, dtype=np.float32)  # Default size


def parse_pascal_voc_xml(xml_file: str) -> List[Dict]:
    """
    Parse PASCAL VOC format XML annotation file.
    
    Parameters:
    -----------
    xml_file : str
        Path ke file XML annotation
        
    Returns:
    --------
    List[Dict]
        List dictionary berisi informasi objek: [{'name': 'A', 'bbox': (x, y, w, h)}, ...]
    """
    objects = []
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            
            # Convert to (x, y, width, height) format
            x, y = xmin, ymin
            w, h = xmax - xmin, ymax - ymin
            
            objects.append({
                'name': name,
                'bbox': (x, y, w, h)
            })
            
    except Exception as e:
        print(f"Error parsing XML file {xml_file}: {e}")
        
    return objects


def load_dataset_from_folders(use_fnt_dataset: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset dari folder structure (tanpa XML annotations).
    Mendukung dua format:
    1. Original format: folder A/, B/, C/, etc. 
    2. FNT format: folder A_upper/, a_lower/, 0_digit/, etc. (case-safe)
    
    Parameters:
    -----------
    use_fnt_dataset : bool
        True untuk menggunakan format FNT (case-safe), False untuk format original
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]        (features, labels) - Features dan labels dalam format numpy array
    """
    features_list = []
    labels_list = []
    
    print("Loading dataset from folder structure...")
    print("Supporting 62 character classes: 0-9, A-Z, a-z")
    
    if use_fnt_dataset:
        # Format FNT dengan case-safe naming
        valid_folders = (
            [f"{i}_digit" for i in range(10)] +                    # 0_digit - 9_digit
            [f"{chr(i)}_upper" for i in range(ord('A'), ord('Z') + 1)] +  # A_upper - Z_upper
            [f"{chr(i)}_lower" for i in range(ord('a'), ord('z') + 1)]    # a_lower - z_lower
        )
        # Mapping dari folder name ke character label
        folder_to_char = {}
        for i in range(10):
            folder_to_char[f"{i}_digit"] = str(i)
        for i in range(ord('A'), ord('Z') + 1):
            folder_to_char[f"{chr(i)}_upper"] = chr(i)
        for i in range(ord('a'), ord('z') + 1):
            folder_to_char[f"{chr(i)}_lower"] = chr(i)
    else:
        # Format original (A-Z, a-z, 0-9) - mendukung 62 karakter
        valid_folders = (
            [chr(i) for i in range(ord('A'), ord('Z') + 1)] +  # A-Z (uppercase)
            [chr(i) for i in range(ord('a'), ord('z') + 1)] +  # a-z (lowercase)
            [str(i) for i in range(10)]                        # 0-9 (digits)
        )
        # Mapping dari folder name ke character label (identity mapping)
        folder_to_char = {folder: folder for folder in valid_folders}    
    print(f"Supporting {len(valid_folders)} character classes")
    if use_fnt_dataset:
        print("Using FNT case-safe format (e.g., A_upper, a_lower, 0_digit)")
        dataset_path = FNT_DATASET_PATH
        print(f"Dataset path: {dataset_path}")
    else:
        print(f"Using original format: {valid_folders}")
        dataset_path = IMAGES_PATH
        print(f"Dataset path: {dataset_path}")
    
    for folder_name in valid_folders:
        folder_path = os.path.join(dataset_path, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} tidak ditemukan, skip...")
            continue
            
        # Cari semua file gambar dalam folder
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        print(f"Processing folder {folder_name}: {len(image_files)} images")
        
        # Get character label from folder name
        char_label = folder_to_char[folder_name]
        
        for img_file in image_files:
            try:
                # Load image
                image = cv2.imread(img_file)
                if image is None:
                    print(f"Warning: Cannot load image {img_file}")
                    continue
                
                # Preprocess entire image sebagai ROI
                preprocessed = preprocess_character_image(image)
                
                # Extract features
                features = extract_features_from_character(preprocessed)
                
                features_list.append(features)
                labels_list.append(char_label)  # Use character label instead of folder name
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
    
    print(f"Total samples loaded: {len(features_list)}")
    
    if len(features_list) == 0:
        print("No data loaded! Please check your dataset structure.")
        return np.array([]), np.array([])
    
    return np.array(features_list), np.array(labels_list)


def load_dataset_with_annotations() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset dengan XML annotations (PASCAL VOC format).
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (features, labels) - Features dan labels dalam format numpy array
    """
    features_list = []
    labels_list = []
    
    print("Loading dataset with XML annotations...")
    
    if not os.path.exists(ANNOTATIONS_PATH):
        print(f"Annotations folder {ANNOTATIONS_PATH} tidak ditemukan!")
        return np.array([]), np.array([])
    
    # Cari semua file XML annotation
    xml_files = glob.glob(os.path.join(ANNOTATIONS_PATH, "*.xml"))
    
    print(f"Found {len(xml_files)} annotation files")
    
    for xml_file in xml_files:
        try:
            # Parse XML annotation
            objects = parse_pascal_voc_xml(xml_file)
            
            if not objects:
                continue
                
            # Cari corresponding image file
            base_name = os.path.splitext(os.path.basename(xml_file))[0]
            
            # Coba berbagai ekstensi
            image_file = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                potential_path = os.path.join(IMAGES_PATH, base_name + ext)
                if os.path.exists(potential_path):
                    image_file = potential_path
                    break
            
            if image_file is None:
                print(f"Warning: Image file for {xml_file} not found")
                continue
                
            # Load image
            image = cv2.imread(image_file)
            if image is None:
                print(f"Warning: Cannot load image {image_file}")
                continue
            
            # Process setiap objek dalam annotation
            for obj in objects:
                try:
                    x, y, w, h = obj['bbox']
                    
                    # Extract ROI
                    roi = image[y:y+h, x:x+w]
                    
                    if roi.size == 0:
                        continue
                        
                    # Preprocess ROI
                    preprocessed = preprocess_character_image(roi)
                    
                    # Extract features
                    features = extract_features_from_character(preprocessed)
                    
                    features_list.append(features)
                    labels_list.append(obj['name'])
                    
                except Exception as e:
                    print(f"Error processing object in {xml_file}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue
    
    print(f"Total samples loaded: {len(features_list)}")
    
    if len(features_list) == 0:
        print("No data loaded! Please check your dataset and annotations.")
        return np.array([]), np.array([])
    
    return np.array(features_list), np.array(labels_list)


def load_dataset_and_extract_features(use_fnt_dataset: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset dan extract features.
    Akan mencoba XML annotations dulu, jika tidak ada akan gunakan folder structure.
    
    Parameters:
    -----------
    use_fnt_dataset : bool
        True untuk menggunakan format FNT (case-safe), False untuk format original
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (features, labels) - Features dan labels dalam format numpy array
    """
    print("Starting dataset loading and feature extraction...")
    
    # Cek apakah ada XML annotations
    if os.path.exists(ANNOTATIONS_PATH) and len(glob.glob(os.path.join(ANNOTATIONS_PATH, "*.xml"))) > 0:
        print("XML annotations found, using annotated dataset...")
        return load_dataset_with_annotations()
    else:
        print("No XML annotations found, using folder structure...")
        return load_dataset_from_folders(use_fnt_dataset=use_fnt_dataset)


# ==================== MAIN TRAINING SCRIPT ====================

def main(use_fnt_dataset: bool = True):
    """
    Main function untuk training alphabetic classifier.
    
    Parameters:
    -----------
    use_fnt_dataset : bool
        True untuk menggunakan format FNT (case-safe), False untuk format original
    """
    import sys
    print("=" * 60)
    print("ALPHABETIC RECOGNITION - TRAINING SCRIPT")
    print("=" * 60)
    
    # Add dataset format information
    if use_fnt_dataset:
        print("📂 Dataset Format: FNT case-safe format")
        print("   Expected folders: 0_digit, 1_digit, ..., A_upper, a_lower, etc.")
    else:
        print("📂 Dataset Format: Original format")
        print("   Expected folders: A, B, C, ..., a, b, c, ..., 0, 1, 2, etc.")
    
    sys.stdout.flush()
    
    # 1. Load dataset dan extract features
    print("\nLoading dataset and extracting features...")
    sys.stdout.flush()
    features, labels = load_dataset_and_extract_features(use_fnt_dataset=use_fnt_dataset)
    
    if len(features) == 0:
        print("Error: No data loaded. Please check your dataset structure.")
        print("\nExpected structure:")
        print("dataset/alphabets/A/ - contains images of letter A")
        print("dataset/alphabets/B/ - contains images of letter B")
        print("... etc ...")
        print("OR")
        print("dataset/alphabets/annotations/ - contains XML annotation files")
        sys.stdout.flush()
        return
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(features)}")
    print(f"Feature vector size: {features.shape[1]}")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Labels distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} samples")
    
    # 2. Split dataset menjadi training dan testing
    print(f"\nSplitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # 3. Initialize dan train classifier
    print(f"\nTraining classifier...")
    
    # Try SVM first
    print("Training SVM classifier...")
    svm_classifier = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42
    )
    svm_classifier.fit(X_train, y_train)
    
    # Predict dan evaluate SVM
    y_pred_svm = svm_classifier.predict(X_test)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    print(f"SVM Accuracy: {svm_accuracy:.4f}")
    
    # Try Random Forest
    print("Training Random Forest classifier...")
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_classifier.fit(X_train, y_train)
    
    # Predict dan evaluate Random Forest
    y_pred_rf = rf_classifier.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # Pilih model terbaik
    if svm_accuracy >= rf_accuracy:
        best_model = svm_classifier
        best_accuracy = svm_accuracy
        best_predictions = y_pred_svm
        model_name = "SVM"
    else:
        best_model = rf_classifier
        best_accuracy = rf_accuracy
        best_predictions = y_pred_rf
        model_name = "Random Forest"
    
    print(f"\nBest model: {model_name} with accuracy: {best_accuracy:.4f}")
    
    # 4. Detailed evaluation
    print(f"\n" + "=" * 60)
    print(f"DETAILED EVALUATION - {model_name}")
    print(f"=" * 60)
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, best_predictions))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, best_predictions))
      # 5. Save model
    print(f"\nSaving model...")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Save model
    model_data = {
        'model': best_model,
        'model_type': model_name,
        'accuracy': best_accuracy,
        'feature_size': features.shape[1],
        'classes': np.unique(labels).tolist(),
        'preprocessing_config': {
            'target_size': CHAR_IMAGE_SIZE,
            'use_otsu': True,
            'use_morphology': True
        }
    }
    
    joblib.dump(model_data, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    
    # 6. Save feature extractor configuration
    feature_config = {
        'target_size': CHAR_IMAGE_SIZE,
        'feature_types': ['hu_moments', 'hog', 'geometric', 'projection', 'zoning', 'crossing'],
        'feature_size': features.shape[1],
        'feature_weights': FEATURE_WEIGHTS,
        'hog_params': HOG_PARAMS
    }
    
    joblib.dump(feature_config, FEATURE_CONFIG_PATH)
    print(f"Feature extractor config saved to: {FEATURE_CONFIG_PATH}")
    
    print(f"\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Final Model: {model_name}")
    print(f"Final Accuracy: {best_accuracy:.4f}")
    print(f"Model Path: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Train Alphabetic Character Recognition Model')
    parser.add_argument('--use-fnt', action='store_true', default=True,
                        help='Use FNT case-safe format (default: True)')
    parser.add_argument('--use-original', action='store_true', 
                        help='Use original format (A/, a/, etc.) - will override --use-fnt')
    
    args = parser.parse_args()
    
    # Determine which format to use
    use_fnt_dataset = args.use_fnt and not args.use_original
    
    main(use_fnt_dataset=use_fnt_dataset)
