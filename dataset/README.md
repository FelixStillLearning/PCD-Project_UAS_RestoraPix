# Dataset Alphabetic Recognition

## Struktur Folder

```
dataset/
└── alphabets/
    ├── annotations/           # File XML anotasi (PASCAL VOC format)
    ├── A/                    # Gambar huruf A
    ├── B/                    # Gambar huruf B
    ├── ...                   # dst untuk C-Z
    ├── 0/                    # Gambar angka 0
    ├── 1/                    # Gambar angka 1
    └── ...                   # dst untuk 2-9
```

## Format Dataset

### Gambar Training
- **Format**: `.jpg`, `.png`, `.bmp`
- **Resolusi**: Minimal 32x32 pixel
- **Background**: Preferably white/light background
- **Orientasi**: Upright characters

### Anotasi (Opsional)
- **Format**: PASCAL VOC XML
- **Tools**: Dapat dibuat menggunakan LabelImg
- **Content**: Bounding box untuk setiap karakter

## Penggunaan

1. Letakkan gambar training di folder yang sesuai (A/, B/, C/, dst.)
2. Minimal 50-100 gambar per karakter untuk hasil yang baik
3. Pastikan variasi:
   - Font yang berbeda
   - Ukuran yang berbeda
   - Slight rotation
   - Noise level yang bervariasi

## Contoh Nama File
```
A/a_001.jpg
A/a_002.png
A/A_arial_001.jpg
B/b_times_001.jpg
0/zero_001.jpg
```
