# Image Processing Application

[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5.5-brightgreen.svg)](https://opencv.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15.11-orange.svg)](https://www.riverbankcomputing.com/software/pyqt/)

Aplikasi pengolahan citra komprehensif dengan antarmuka grafis berbasis PyQt5. Aplikasi ini dapat melakukan berbagai operasi pengolahan citra seperti operasi dasar, filtering, deteksi tepi, operasi morfologi, segmentasi, dan transformasi geometris.

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
│   └── transformation/     # Transformasi geometris
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
4. **Pemrosesan Dasar**: Gunakan menu atau tombol untuk menerapkan berbagai operasi pemrosesan citra

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

## Lisensi

Proyek ini dilisensikan di bawah lisensi MIT - lihat file [LICENSE](LICENSE) untuk detailnya.

## Kontributor

- FelixStillLearning (felixangga077@email.com)

## Referensi

- [OpenCV Documentation](https://docs.opencv.org/)
- [Digital Image Processing - Gonzalez & Woods](https://www.pearson.com/en-us/subject-catalog/p/digital-image-processing/P200000003546/9780137358144)
