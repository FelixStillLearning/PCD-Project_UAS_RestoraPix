# GitHub Repository Setup Guide

## 📋 What's Included in This Repository

This repository contains the **Alphabetic Recognition System** - a complete machine learning project for recognizing 62 character classes (0-9, A-Z, a-z) from images.

### ✅ Included Files:
- **Source Code**: All Python modules and scripts
- **Documentation**: Comprehensive guides and README files
- **Configuration**: Requirements.txt and setup files
- **Project Structure**: Organized folder structure
- **Sample Images**: Small example images in `/assets/` and `/screenshots/`
- **Model Architecture**: Code for building models (not trained weights)

### ❌ Excluded Files (due to size):
- **Trained Models**: `.pkl`, `.pt`, `.h5` files (too large for GitHub)
- **Datasets**: Raw image datasets (62,992 PNG files)
- **YOLO Weights**: Pre-trained YOLO model files
- **Training Outputs**: Logs, checkpoints, evaluation results

## 🚀 Quick Setup for New Users

### 1. Clone Repository
```bash
git clone <your-repository-url>
cd Project_UAS
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required Files

#### A. Download Chars74K Dataset
1. Visit: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
2. Download "English/Fnt" subset
3. Extract to: `dataset/alphabets/`

#### B. Download YOLO Weights (Optional)
```bash
# For object detection features
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
```

### 5. Organize Dataset
```bash
# Reorganize dataset for training
python reorganize_fnt_dataset.py
```

### 6. Train Model (Optional)
```bash
# Train your own model
python train_alphabet_classifier.py --use-fnt
```

## 📁 Repository Structure

```
Project_UAS/
├── 📂 core/                    # Core functionality
├── 📂 modules/                 # Recognition modules
├── 📂 gui/                     # GUI components
├── 📂 utils/                   # Utility functions
├── 📂 assets/                  # Sample images (small)
├── 📂 screenshots/             # Documentation images
├── 📂 dataset/                 # Dataset structure (empty)
│   ├── alphabets/              # Original Chars74K location
│   └── fnt_organized/          # Reorganized dataset location
├── 📂 models/                  # Model architecture (no weights)
├── 📄 app.py                   # Main application
├── 📄 train_alphabet_classifier.py  # Training script
├── 📄 reorganize_fnt_dataset.py     # Dataset organization
├── 📄 requirements.txt         # Dependencies
├── 📄 README.md               # Main documentation
└── 📄 .gitignore              # Git ignore rules
```

## 🔧 Configuration

### Dataset Paths
Update paths in `modules/alphabetic_recognition/config.py`:
```python
# Update these paths based on your setup
DATASET_PATH = "dataset/alphabets"
FNT_DATASET_PATH = "dataset/fnt_organized"
MODEL_PATH = "models/"
```

### Model Training Options
```python
# Use original Sample folders
python train_alphabet_classifier.py --use-original

# Use reorganized case-safe folders (recommended)
python train_alphabet_classifier.py --use-fnt
```

## 📊 Expected Dataset Structure

### After downloading Chars74K:
```
dataset/alphabets/
├── Sample001/  # Digit '0' (1,016 images)
├── Sample002/  # Digit '1' (1,016 images)
├── ...
├── Sample062/  # Letter 'z' (1,016 images)
```

### After reorganization:
```
dataset/fnt_organized/
├── 0_digit/    # Digit '0' (1,016 images)
├── 1_digit/    # Digit '1' (1,016 images)
├── ...
├── A_upper/    # Letter 'A' (1,016 images)
├── ...
├── a_lower/    # Letter 'a' (1,016 images)
├── ...
├── z_lower/    # Letter 'z' (1,016 images)
```

## 🧪 Testing

Run the complete test suite:
```bash
python test_production_system.py
```

Run specific tests:
```bash
python test_model.py
python test_integration.py
```

## 📚 Documentation

- `README.md` - Main project overview
- `TRAINING_GUIDE.md` - Detailed training instructions
- `PRODUCTION_READY_SUMMARY.md` - System capabilities
- `INTEGRATION_COMPLETE.md` - Integration details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Run tests
5. Submit pull request

## 📞 Support

If you encounter issues:
1. Check the documentation files
2. Ensure all dependencies are installed
3. Verify dataset structure matches expected format
4. Run test suite to validate setup

## 📄 License

See `LICENSE` file for details.

---

**Note**: This project requires significant disk space (~2GB) for the complete dataset. The GitHub repository contains only the code structure to keep it lightweight.
