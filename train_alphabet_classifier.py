#!/usr/bin/env python3
"""
Alphabetic Recognition Training Script
=====================================

Skrip untuk ekstraksi fitur dan pelatihan model klasifikasi karakter alfanumerik
menggunakan teknik Pengolahan Citra Digital (PCD) klasik dan Machine Learning klasik.

Author: [Your Name]
Date: May 2025
"""

import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import glob
from typing import Tuple, List, Dict, Union

# Import modul PCD yang sudah ada
try:
    from modules.color_processing.color_processor import to_grayscale
    from modules.filtering.filters import gaussian_filter, median_filter
    from modules.morphology.operations import opening, closing, erosion, dilation
    from modules.segmentation.thresholding import otsu_threshold, adaptive_threshold
    from modules.transformation.geometric import resize_image
    print("✓ PCD modules imported successfully")
except ImportError as e:
    print(f"⚠ Warning: Could not import some PCD modules: {e}")
    print("  Script will use OpenCV fallbacks where necessary")


# ==================== DEFINISI PATH ====================
DATASET_PATH = "dataset/alphabets"
IMAGES_PATH = os.path.join(DATASET_PATH)
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "annotations")
MODEL_PATH = "models/alphabetic_classifier_model.pkl"
FEATURE_EXTRACTOR_PATH = "models/feature_extractor_config.pkl"

# Target ukuran untuk normalisasi karakter
TARGET_SIZE = (28, 28)

# ==================== PREPROCESSING FUNCTIONS ====================

def preprocess_character_image(image_roi: np.ndarray) -> np.ndarray:
    """
    Preprocess ROI karakter untuk ekstraksi fitur.
    
    Parameters:
    -----------
    image_roi : np.ndarray
        ROI dari gambar karakter yang sudah tersegmen
        
    Returns:
    --------
    np.ndarray
        Gambar biner karakter yang sudah bersih dan dinormalisasi ukurannya
    """
    try:
        # 1. Konversi ke grayscale menggunakan modul yang sudah ada
        if len(image_roi.shape) == 3:
            try:
                gray_image = to_grayscale(image_roi)
            except:
                gray_image = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image_roi.copy()
        
        # 2. Noise reduction menggunakan Gaussian blur
        try:
            denoised = gaussian_filter(gray_image, filter_size=3)
        except:
            denoised = cv2.GaussianBlur(gray_image, (3, 3), 1.0)
        
        # 3. Median filter untuk menghilangkan salt-and-pepper noise
        try:
            denoised = median_filter(denoised, filter_size=3)
        except:
            denoised = cv2.medianBlur(denoised, 3)
        
        # 4. Binarization menggunakan Otsu thresholding
        try:
            _, binary_image = otsu_threshold(denoised)
        except:
            _, binary_image = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. Morphological operations untuk pembersihan
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        try:
            # Opening untuk menghilangkan noise kecil
            cleaned = opening(binary_image, kernel)
            # Closing untuk mengisi lubang kecil
            cleaned = closing(cleaned, kernel)
        except:
            cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # 6. Normalisasi ukuran ke TARGET_SIZE
        try:
            normalized = resize_image(cleaned, TARGET_SIZE)
        except:
            normalized = cv2.resize(cleaned, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        # 7. Pastikan background hitam (0) dan foreground putih (255)
        # Jika lebih banyak pixel putih, invert
        white_pixels = np.count_nonzero(normalized == 255)
        total_pixels = normalized.shape[0] * normalized.shape[1]
        
        if white_pixels > total_pixels * 0.5:
            normalized = cv2.bitwise_not(normalized)
        
        return normalized
        
    except Exception as e:
        print(f"Error in preprocess_character_image: {e}")
        # Return default processed image using OpenCV only
        if len(image_roi.shape) == 3:
            gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_roi.copy()
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 1.0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        normalized = cv2.resize(cleaned, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        return normalized


def extract_features_from_character(binary_char_image: np.ndarray) -> np.ndarray:
    """
    Ekstrak fitur dari gambar biner karakter yang sudah dinormalisasi.
    
    Parameters:
    -----------
    binary_char_image : np.ndarray
        Gambar biner karakter yang sudah dinormalisasi (28x28)
        
    Returns:
    --------
    np.ndarray
        Vektor fitur NumPy
    """
    features = []
    
    try:
        # 1. Hu Moments - Sangat penting untuk bentuk karakter
        moments = cv2.moments(binary_char_image)
        hu_moments = cv2.HuMoments(moments)
        # Log transform untuk stabilitas numerik
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        features.extend(hu_moments.flatten())
        
        # 2. HOG Features - Histogram of Oriented Gradients
        # Konfigurasi HOG untuk karakter kecil
        hog = cv2.HOGDescriptor(
            _winSize=(28, 28),
            _blockSize=(14, 14),
            _blockStride=(7, 7),
            _cellSize=(7, 7),
            _nbins=9
        )
        hog_features = hog.compute(binary_char_image)
        features.extend(hog_features.flatten())
        
        # 3. Fitur Geometris
        # Aspect ratio
        contours, _ = cv2.findContours(binary_char_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            features.append(aspect_ratio)
            
            # Area ratio (area karakter / area bounding box)
            char_area = cv2.contourArea(largest_contour)
            bbox_area = w * h
            area_ratio = char_area / bbox_area if bbox_area > 0 else 0
            features.append(area_ratio)
            
            # Perimeter to area ratio
            perimeter = cv2.arcLength(largest_contour, True)
            perimeter_area_ratio = perimeter / char_area if char_area > 0 else 0
            features.append(perimeter_area_ratio)
            
            # Solidity (area / convex hull area)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = char_area / hull_area if hull_area > 0 else 0
            features.append(solidity)
            
            # Extent (area / bounding box area)
            extent = char_area / bbox_area if bbox_area > 0 else 0
            features.append(extent)
        else:
            # Jika tidak ada contour, berikan nilai default
            features.extend([0, 0, 0, 0, 0])
        
        # 4. Fitur Proyeksi - Horizontal dan Vertical Projection
        h_projection = np.sum(binary_char_image, axis=1)  # Horizontal projection
        v_projection = np.sum(binary_char_image, axis=0)  # Vertical projection
        
        # Statistik proyeksi
        features.extend([
            np.mean(h_projection), np.std(h_projection), np.max(h_projection),
            np.mean(v_projection), np.std(v_projection), np.max(v_projection)
        ])
        
        # 5. Zoning Features - Bagi image menjadi 4x4 zona
        zones = []
        zone_h, zone_w = binary_char_image.shape[0] // 4, binary_char_image.shape[1] // 4
        for i in range(4):
            for j in range(4):
                zone = binary_char_image[i*zone_h:(i+1)*zone_h, j*zone_w:(j+1)*zone_w]
                zone_density = np.sum(zone) / (zone_h * zone_w * 255)  # Normalize
                zones.append(zone_density)
        features.extend(zones)
        
        # 6. Crossing Features - Hitung crossing points
        # Horizontal crossings
        h_crossings = 0
        for row in binary_char_image:
            crossings = 0
            for i in range(len(row) - 1):
                if row[i] != row[i + 1]:
                    crossings += 1
            h_crossings += crossings
        
        # Vertical crossings
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
        print(f"Error in extract_features_from_character: {e}")
        # Return default feature vector
        return np.zeros(100, dtype=np.float32)  # Adjust size as needed


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


def load_dataset_from_folders() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset dari folder structure (tanpa XML annotations).
    Setiap folder (A/, B/, C/, etc.) berisi gambar untuk karakter tersebut.
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (features, labels) - Features dan labels dalam format numpy array
    """
    features_list = []
    labels_list = []
    
    print("Loading dataset from folder structure...")
    
    # List folder yang valid (A-Z, 0-9)
    valid_folders = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [str(i) for i in range(10)]
    
    for folder_name in valid_folders:
        folder_path = os.path.join(IMAGES_PATH, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} tidak ditemukan, skip...")
            continue
            
        # Cari semua file gambar dalam folder
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        print(f"Processing folder {folder_name}: {len(image_files)} images")
        
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
                labels_list.append(folder_name)
                
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


def load_dataset_and_extract_features() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset dan extract features.
    Akan mencoba XML annotations dulu, jika tidak ada akan gunakan folder structure.
    
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
        return load_dataset_from_folders()


# ==================== MAIN TRAINING SCRIPT ====================

def main():
    """
    Main function untuk training alphabetic classifier.
    """
    import sys
    print("=" * 60)
    print("ALPHABETIC RECOGNITION - TRAINING SCRIPT")
    print("=" * 60)
    sys.stdout.flush()
    
    # 1. Load dataset dan extract features
    print("Loading dataset and extracting features...")
    sys.stdout.flush()
    features, labels = load_dataset_and_extract_features()
    
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
    
    # Buat folder models jika belum ada
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Save model
    model_data = {
        'model': best_model,
        'model_type': model_name,
        'accuracy': best_accuracy,
        'feature_size': features.shape[1],
        'classes': np.unique(labels).tolist(),
        'preprocessing_config': {
            'target_size': TARGET_SIZE,
            'use_otsu': True,
            'use_morphology': True
        }
    }
    
    joblib.dump(model_data, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    
    # 6. Simpan juga konfigurasi feature extractor
    feature_config = {
        'target_size': TARGET_SIZE,
        'feature_types': ['hu_moments', 'hog', 'geometric', 'projection', 'zoning', 'crossing'],
        'feature_size': features.shape[1]
    }
    
    joblib.dump(feature_config, FEATURE_EXTRACTOR_PATH)
    print(f"Feature extractor config saved to: {FEATURE_EXTRACTOR_PATH}")
    
    print(f"\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Final Model: {model_name}")
    print(f"Final Accuracy: {best_accuracy:.4f}")
    print(f"Model Path: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
