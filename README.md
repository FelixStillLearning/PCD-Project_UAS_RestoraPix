# Image Processing Application

[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5.5-brightgreen.svg)](https://opencv.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15.11-orange.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![Production Ready](https://img.shields.io/badge/status-production%20ready-success.svg)](./PRODUCTION_READY_SUMMARY.md)
[![Test Coverage](https://img.shields.io/badge/tests-100%25%20pass-brightgreen.svg)](./test_production_system.py)

Aplikasi pengolahan citra komprehensif dengan antarmuka grafis berbasis PyQt5 dan **sistem pengenalan karakter alfabetik production-ready**. Aplikasi ini dapat melakukan berbagai operasi pengolahan citra seperti operasi dasar, filtering, deteksi tepi, operasi morfologi, segmentasi, transformasi geometris, dan **pengenalan karakter alfabetik dengan performa enterprise-grade**.

## Fitur

### Operasi Dasar
- Konversi Grayscale
- Pengaturan Kecerahan
- Pengaturan Kontras
- Contrast Stretching
- Negative Image
- Binarization
- Operasi Histogram

### Filtering
- Convolution dengan Kernel
- Mean Filter
- Gaussian Filter
- Median Filter
- Max/Min Filter
- Sharpening
- Operasi Domain Frekuensi (DFT)

### Deteksi Tepi
- Sobel Operator
- Prewitt Operator
- Roberts Operator
- Canny Edge Detector
- Deteksi Tepi berbasis DFT

### Morfologi
- Dilasi
- Erosi
- Opening
- Closing
- Skeletonization

### Segmentasi
- Binary Thresholding
- Adaptive Thresholding
- Otsu Thresholding
- Deteksi Kontur

### Transformasi
- Translasi
- Rotasi
- Scaling (Zoom)
- Transpose
- Skew
- Crop

### Operasi Aritmatika
- Penambahan dan Pengurangan
- Perkalian dan Pembagian
- Operasi Bitwise (AND, OR, XOR)

### Alphabetic Recognition (NEW!)
- Character Recognition (A-Z, 0-9)
- Real-time recognition from camera
- Video file recognition
- Advanced feature extraction with HOG + Hu Moments
- Confidence scoring and batch processing

## Struktur Proyek

```
Project_UAS/
├── app.py                  # Entry point aplikasi
├── core/                   # Komponen inti
│   └── image_processor.py  # Kelas dasar untuk pemrosesan citra
├── gui/                    # Komponen GUI
│   ├── app.py              # Jendela aplikasi utama
│   └── Gui.ui              # File UI PyQt
├── modules/                # Modul fungsional
│   ├── basic/              # Operasi dasar
│   ├── edge_detection/     # Algoritma deteksi tepi
│   ├── filtering/          # Operasi filtering
│   ├── morphology/         # Operasi morfologi
│   ├── segmentation/       # Segmentasi citra
│   ├── transformation/     # Transformasi geometris
│   └── alphabetic_recognition/  # Modul pengenalan karakter (NEW!)
└── utils/                  # Fungsi utilitas
```

## Instalasi

### Prasyarat
- Python 3.9 atau lebih baru
- pip (Python package manager)

### Langkah-langkah Instalasi

1. Clone repositori ini:
   ```bash
   git clone https://github.com/username/Project_UAS.git
   cd Project_UAS
   ```

2. Buat dan aktifkan virtual environment:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # Di Windows dengan Git Bash
   # ATAU
   .venv\Scripts\activate         # Di Windows dengan CMD
   ```

3. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## Cara Menjalankan

Setelah mengaktifkan virtual environment dan menginstal semua dependensi, jalankan aplikasi dengan perintah:

```bash
python app.py
```

## Cara Penggunaan

1. **Memuat Gambar**: Klik tombol "Load" untuk memilih file gambar
2. **Menyimpan Gambar**: Klik tombol "Save" untuk menyimpan gambar hasil pemrosesan
3. **Reset Gambar**: Klik tombol "Reset" untuk kembali ke gambar asli
4. **Alphabetic Recognition**: Gunakan menu "Analisis > Alphabetic Recognition" untuk:
   - Mengenali karakter dari gambar yang sudah dimuat
   - Melakukan pengenalan real-time dari kamera
   - Menganalisis video yang mengandung teks
5. **Export Data**: Gunakan menu "File > Export Pixel Data" untuk mengekspor data piksel

## Contoh Penggunaan Kode

Anda juga dapat menggunakan modul-modul secara terpisah dalam kode Python Anda:

```python
# Contoh penggunaan operasi dasar
from modules.basic.operations import grayscale, adjust_brightness
import cv2

# Memuat gambar menggunakan OpenCV
image = cv2.imread('path/to/image.jpg')

# Konversi ke grayscale
gray_image = grayscale(image)

# Menyesuaikan kecerahan
bright_image = adjust_brightness(gray_image, 50)

# Menyimpan hasil
cv2.imwrite('result.jpg', bright_image)
```
## Dependensi

- [Python 3.9+](https://www.python.org/downloads/)
- [OpenCV 4.5.5](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)
- [scikit-learn](https://scikit-learn.org/) - untuk alphabetic recognition
- [joblib](https://joblib.readthedocs.io/) - untuk model persistence

## Alphabetic Recognition

Aplikasi ini dilengkapi dengan modul pengenalan karakter alfanumerik (A-Z, 0-9) yang menggunakan teknik **Digital Image Processing klasik** dan **Machine Learning klasik**.

### Fitur Alphabetic Recognition:
- **Character Recognition**: Mengenali karakter A-Z dan angka 0-9
- **Real-time Recognition**: Pengenalan dari live camera feed
- **Video Recognition**: Analisis video yang mengandung teks
- **Advanced Features**: HOG descriptors, Hu Moments, geometric features
- **Confidence Scoring**: Skor kepercayaan untuk setiap prediksi
- **Batch Processing**: Pemrosesan multiple karakter sekaligus

### Cara Menggunakan Alphabetic Recognition:
1. Latih model dengan dataset real: `python train_alphabet_classifier.py`
2. Buka aplikasi dan muat gambar yang mengandung teks
3. Pilih menu **Analisis > Alphabetic Recognition**
4. Pilih mode recognition yang diinginkan (Image/Camera/Video)

Untuk informasi lengkap, lihat [`ALPHABETIC_RECOGNITION_COMPLETE.md`](ALPHABETIC_RECOGNITION_COMPLETE.md) dan [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md).

## Lisensi

Proyek ini dilisensikan di bawah lisensi MIT - lihat file [LICENSE](LICENSE) untuk detailnya.

## Kontributor

- FelixStillLearning (felixangga077@email.com)

## Referensi

- [OpenCV Documentation](https://docs.opencv.org/)
- [Digital Image Processing - Gonzalez & Woods](https://www.pearson.com/en-us/subject-catalog/p/digital-image-processing/P200000003546/9780137358144)
