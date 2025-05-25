# Alphabetic Recognition Training Guide

## Overview
Training script untuk klasifikasi karakter alfanumerik (A-Z, 0-9) menggunakan teknik Digital Image Processing (PCD) klasik dan Machine Learning klasik.

## Features
- **Centralized Configuration**: All parameters managed in `config.py` for consistency
- **Enhanced Preprocessing**: Optimized pipeline using existing PCD modules
- **Weighted Feature Extraction**: Hu Moments, HOG, geometric features, projection features, zoning, dan crossing points
- **Advanced Classification**: SVM dan Random Forest dengan optimized parameters
- **Dataset Support**: Folder structure atau XML annotations (PASCAL VOC format)
- **Performance Optimization**: Batch processing dan caching mechanisms
- **Comprehensive Testing**: Full test coverage dengan performance benchmarking

## Dataset Structure

### Option 1: Folder Structure
```
dataset/alphabets/
├── A/              # Images of letter A
│   ├── A_01.jpg
│   ├── A_02.png
│   └── ...
├── B/              # Images of letter B
│   ├── B_01.jpg
│   └── ...
├── ...
├── Z/              # Images of letter Z
├── 0/              # Images of digit 0
├── 1/              # Images of digit 1
├── ...
└── 9/              # Images of digit 9
```

### Option 2: XML Annotations (PASCAL VOC Format)
```
dataset/alphabets/
├── image001.jpg    # Original images
├── image002.jpg
├── ...
└── annotations/    # XML annotation files
    ├── image001.xml
    ├── image002.xml
    └── ...
```

XML Format:
```xml
<annotation>
    <filename>image001.jpg</filename>
    <object>
        <name>A</name>
        <bndbox>
            <xmin>10</xmin>
            <ymin>20</ymin>
            <xmax>50</xmax>
            <ymax>60</ymax>
        </bndbox>
    </object>
    <object>
        <name>B</name>
        <bndbox>
            <xmin>60</xmin>
            <ymin>20</ymin>
            <xmax>100</xmax>
            <ymax>60</ymax>
        </bndbox>
    </object>
</annotation>
```

## Usage

### 1. Quick Test with Dummy Data
```bash
# Create dummy dataset for testing
python create_dummy_dataset.py

# Run training
python train_alphabet_classifier.py
```

### 2. Training with Real Dataset
1. Prepare your dataset in one of the formats above
2. Place images and/or annotations in the correct folders
3. Run training:
```bash
python train_alphabet_classifier.py
```

### 3. Training Output
```
=============================================================
ALPHABETIC RECOGNITION - TRAINING SCRIPT
=============================================================

Loading dataset and extracting features...
✓ Dataset loaded successfully:
  - Total samples: 30
  - Feature dimensions: 100+
  - Number of classes: 6
  - Classes: ['0', '1', '2', 'A', 'B', 'C']

Class distribution:
  0: 5 samples
  1: 5 samples
  2: 5 samples
  A: 5 samples
  B: 5 samples
  C: 5 samples

Splitting dataset (80% train, 20% test)...
  - Training samples: 24
  - Testing samples: 6

Training classifier...
Training SVM classifier...
SVM Accuracy: 0.8333
Training Random Forest classifier...
Random Forest Accuracy: 1.0000

Best model: Random Forest with accuracy: 1.0000

=============================================================
DETAILED EVALUATION - Random Forest
=============================================================

Accuracy: 1.0000

Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00         1
           2       1.00      1.00      1.00         1
           A       1.00      1.00      1.00         1
           B       1.00      1.00      1.00         1
           C       1.00      1.00      1.00         1

    accuracy                           1.00         6
   macro avg       1.00      1.00      1.00         6
weighted avg       1.00      1.00      1.00         6

Confusion Matrix:
[[1 0 0 0 0 0]
 [0 1 0 0 0 0]
 [0 0 1 0 0 0]
 [0 0 0 1 0 0]
 [0 0 0 0 1 0]
 [0 0 0 0 0 1]]

Saving model...
✓ Model saved to: models/alphabetic_classifier_model.pkl
Feature extractor config saved to: models/feature_extractor_config.pkl

=============================================================
TRAINING COMPLETED SUCCESSFULLY!
Final Model: Random Forest
Final Accuracy: 1.0000
Model Path: models/alphabetic_classifier_model.pkl
=============================================================
```

## Output Files

### Model Files
- `models/alphabetic_classifier_model.pkl`: Trained model with metadata
- `models/feature_extractor_config.pkl`: Feature extraction configuration

### Model Data Structure
```python
model_data = {
    'model': trained_classifier,           # Scikit-learn model object
    'model_type': 'Random Forest',         # Model type name
    'accuracy': 1.0000,                    # Test accuracy
    'feature_size': 100,                   # Feature vector size
    'classes': ['A', 'B', 'C', ...],      # List of character classes
    'preprocessing_config': {              # Preprocessing parameters
        'target_size': (28, 28),
        'use_otsu': True,
        'use_morphology': True
    }
}
```

## Feature Extraction Details

### 1. Hu Moments (7 features)
- Translation, rotation, and scale invariant shape descriptors
- Log-transformed for numerical stability

### 2. HOG Features (~36 features)
- Histogram of Oriented Gradients
- Window size: 28x28, Block size: 14x14, Cell size: 7x7

### 3. Geometric Features (5 features)
- Aspect ratio
- Area ratio (contour area / bounding box area)
- Perimeter to area ratio
- Solidity (area / convex hull area)
- Extent (area / bounding box area)

### 4. Projection Features (6 features)
- Horizontal and vertical projection statistics
- Mean, standard deviation, and maximum values

### 5. Zoning Features (16 features)
- Image divided into 4x4 zones
- Pixel density calculated for each zone

### 6. Crossing Features (2 features)
- Horizontal and vertical crossing point counts
- Measures structural complexity

**Total Feature Vector Size**: ~360 features (optimized configuration)

## Configuration Management

The alphabetic recognition system uses centralized configuration management located in `modules/alphabetic_recognition/config.py`. This provides:

### Key Configuration Features:
- **Centralized Parameters**: All HOG, preprocessing, and model parameters in one place
- **Optimized Settings**: Fine-tuned parameters for character recognition
- **Validation Functions**: Automatic configuration validation
- **Path Management**: Automatic directory creation and path resolution
- **Backward Compatibility**: Support for existing imports and code

### Configuration Sections:
- **Image Processing**: Target size, filtering parameters, morphological operations
- **Feature Extraction**: HOG parameters, feature weights, ensemble configuration
- **Model Parameters**: SVM and Random Forest hyperparameters
- **Performance Settings**: Batch size, multiprocessing, caching options

### Accessing Configuration:
```python
from modules.alphabetic_recognition.config import (
    get_preprocessing_config,
    get_feature_extraction_config, 
    get_classification_config,
    validate_config
)

# Load configurations
preprocessing_config = get_preprocessing_config()
feature_config = get_feature_extraction_config()
model_config = get_classification_config()

# Validate all configurations
validate_config()
```

## Integration with Existing System

### Loading Trained Model
```python
import joblib

# Load trained model
model_data = joblib.load('models/alphabetic_classifier_model.pkl')
classifier = model_data['model']
classes = model_data['classes']

# Load feature extractor config
feature_config = joblib.load('models/feature_extractor_config.pkl')
```

### Prediction Pipeline
```python
from train_alphabet_classifier import preprocess_character_image, extract_features_from_character

def predict_character(image_roi):
    # Preprocess
    preprocessed = preprocess_character_image(image_roi)
    
    # Extract features
    features = extract_features_from_character(preprocessed)
    
    # Predict
    prediction = classifier.predict([features])[0]
    confidence = classifier.predict_proba([features]).max()
    
    return prediction, confidence
```

## Tips for Better Performance

### 1. Dataset Quality
- Use clear, high-contrast character images
- Include various fonts and styles
- Ensure balanced dataset (similar number of samples per class)
- Minimum 20-50 samples per character for good performance

### 2. Image Preprocessing
- Images should be properly cropped around characters
- Remove excessive whitespace/background
- Ensure consistent character orientation

### 3. Feature Engineering
- The current feature set works well for printed characters
- For handwritten characters, consider adding more texture features
- Adjust HOG parameters for different character sizes

### 4. Model Selection
- SVM works well with smaller datasets
- Random Forest is more robust to noise
- Consider ensemble methods for production use

## Troubleshooting

### Import Errors
- The script has fallbacks to OpenCV if PCD modules fail to import
- Ensure all dependencies are installed: `pip install -r requirements.txt`

### No Data Loaded
- Check dataset structure matches expected format
- Verify image file extensions (.jpg, .jpeg, .png, .bmp)
- For XML annotations, ensure image files exist

### Low Accuracy
- Increase dataset size
- Check data quality and consistency
- Verify character images are properly cropped
- Consider adjusting preprocessing parameters

### Memory Issues
- For large datasets, consider batch processing
- Reduce HOG descriptor parameters if needed
- Use data augmentation instead of collecting more images
