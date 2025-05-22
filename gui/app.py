# filepath: d:\Development\Proyek\Citra\Project_UAS\gui\app.py
"""
Main GUI application module
Contains the main application class for the image processing GUI
"""
import sys
import os
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt

# Add parent directory to sys.path to enable absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import modules from the parent directory
import cv2
from core.image_processor import ImageProcessor

# Import basic operations modules
from modules.basic.operations import (
    grayscale, adjust_brightness, adjust_contrast, 
    contrast_stretching, negative, binarize
)
from modules.basic.histogram import (
    calculate_grayscale_histogram, calculate_rgb_histogram, 
    calculate_equalized_histogram,
    plot_grayscale_histogram, plot_rgb_histogram, 
    plot_equalized_histogram
)
from modules.basic.arithmetic import (
    add_images, subtract_images, multiply_images, 
    divide_images, bitwise_and, bitwise_or, bitwise_xor
)

# Import filtering modules
from modules.filtering.filters import (
    convolution, mean_filter, gaussian_filter, 
    median_filter, max_filter, min_filter
)
from modules.filtering.sharpening import sharpen, high_pass_filter
from modules.filtering.frequency_domain import (
    dft, low_pass_filter, high_pass_filter, 
    band_pass_filter, plot_frequency_domain_result
)

# Import edge detection modules
from modules.edge_detection.detectors import (
    sobel_edge_detection, prewitt_edge_detection, 
    roberts_edge_detection, canny_edge_detection, 
    dft_edge_detection
)

# Import morphology modules
from modules.morphology.operations import (
    to_binary, dilation, erosion, opening, 
    closing, skeletonize
)

# Import segmentation modules
from modules.segmentation.thresholding import (
    binary_threshold, binary_inv_threshold, trunc_threshold,
    tozero_threshold, tozero_inv_threshold, 
    adaptive_mean_threshold, adaptive_gaussian_threshold,
    otsu_threshold, contour_detection
)

# Import transformation modules
from modules.transformation.geometric import (
    translate, rotate, rotate90, rotate180, rotate270,
    transpose, zoom, zoom_in, zoom_out, skew, crop
)

# Import restoration modules
from modules.restoration.operations import (
    inpainting, deblurring, old_photo_restoration
)

class ImageProcessingApp(QMainWindow):
    """
    Main application class for the image processing GUI
    """
    def __init__(self):
        """Initialize the application"""
        super(ImageProcessingApp, self).__init__()
        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Gui.ui')
        loadUi(ui_path, self)
        
        # Initialize image processor
        self.processor = ImageProcessor()
        
        # Connect UI elements to functions
        self.setup_connections()
    
    def setup_connections(self):
        """Set up connections between UI elements and functions"""
        # File operations
        self.loadButton.clicked.connect(self.load_image)
        self.savebutton.clicked.connect(self.save_image)
        self.resetbutton.clicked.connect(self.reset_image)
        
        # Basic operations
        self.actionoperasi_pencerahan.triggered.connect(self.apply_brightness)
        self.actionOperasi_Kontras.triggered.connect(self.apply_contrast)
        self.actionOperasi_Kontras_Stretching.triggered.connect(self.apply_contrast_stretching)
        self.actionNegative_Image.triggered.connect(self.apply_negative)
        self.actionBiner_Image.triggered.connect(self.apply_binarization)
        self.actionGrayscale_Image.triggered.connect(self.apply_grayscale)
        
        # Histogram operations
        self.actionHistogram_Grayscale.triggered.connect(self.show_gray_histogram)
        self.actionHistogram_RGB.triggered.connect(self.show_rgb_histogram)
        self.actionHistogram_Equalization.triggered.connect(self.apply_histogram_equalization)
        
        # Transformation operations
        self.actiontranslasi.triggered.connect(self.apply_translation)
        self.action90_Derajat.triggered.connect(self.apply_rotate_90)
        self.action_90_Derajat.triggered.connect(self.apply_rotate_neg_90)
        self.action180_Derajat.triggered.connect(self.apply_rotate_180)
        self.action45_Derajat.triggered.connect(self.apply_rotate_45)
        self.action_45_Derajat.triggered.connect(self.apply_rotate_neg_45)
        self.actionTranspose.triggered.connect(self.apply_transpose)
        
        # Zoom operations
        self.action2x.triggered.connect(self.apply_zoom_in_2x)
        self.action3x.triggered.connect(self.apply_zoom_in_3x)
        self.action4x.triggered.connect(self.apply_zoom_in_4x)
        self.action1_2.triggered.connect(self.apply_zoom_out_half)
        self.action1_4.triggered.connect(self.apply_zoom_out_quarter)
        self.action3_4.triggered.connect(self.apply_zoom_out_three_quarters)
        self.actionSkewed.triggered.connect(self.apply_skew)
        self.actionCrop.triggered.connect(self.apply_crop)
        
        # Arithmetic operations
        self.actionTambah_dan_Kurang.triggered.connect(self.apply_add_subtract)
        self.actionKali_dan_Bagi.triggered.connect(self.apply_multiply_divide)
        self.actionOperasi_AND.triggered.connect(self.apply_bitwise_and)
        self.actionOperasi_OR.triggered.connect(self.apply_bitwise_or)
        self.actionOperasi_XOR.triggered.connect(self.apply_bitwise_xor)
        
        # Filtering operations
        self.actionFiltering.triggered.connect(self.apply_filtering)
        self.actionSharpening.triggered.connect(self.apply_sharpening)
        self.actionMedian_Filter.triggered.connect(self.apply_median_filter)
        self.actionMax_Filter.triggered.connect(self.apply_max_filter)
        self.actionMin_Filter.triggered.connect(self.apply_min_filter)
        
        # Frequency domain operations
        self.actionDFT_Smoothing_Image.triggered.connect(self.apply_dft_smoothing)
        self.actionDFT_Edge_Detection.triggered.connect(self.apply_dft_edge_detection)
        
        # Edge detection operations
        self.actionSobel.triggered.connect(self.apply_sobel_edge_detection)
        self.actionPrewitt.triggered.connect(self.apply_prewitt_edge_detection)
        self.actionRobert.triggered.connect(self.apply_robert_edge_detection)
        self.actionCanny.triggered.connect(self.apply_canny_edge_detection)
        
        # Morphological operations
        self.actionDilasi.triggered.connect(self.apply_dilation)
        self.actionErosi.triggered.connect(self.apply_erosion)
        self.actionopening.triggered.connect(self.apply_opening)
        self.actionClosing.triggered.connect(self.apply_closing)
        self.actionSkeletonizing.triggered.connect(self.apply_skeletonizing)
        
        # Thresholding operations
        self.actionBinary_Thresholding.triggered.connect(self.apply_binary_threshold)
        self.actionBinary_Inv_Thresholding.triggered.connect(self.apply_binary_inv_threshold)
        self.actionTrunc_Thresholding.triggered.connect(self.apply_trunc_threshold)
        self.actionToZero_Thresholding.triggered.connect(self.apply_tozero_threshold)
        self.actionToZero_Inv_Thresholding.triggered.connect(self.apply_tozero_inv_threshold)
        self.actionMean_Thresholding.triggered.connect(self.apply_adaptive_mean_threshold)
        self.actionGaussian_Thresholding.triggered.connect(self.apply_adaptive_gaussian_threshold)
        self.actionOtsu_Thresholding.triggered.connect(self.apply_otsu_threshold)
        
        # Contour detection
        self.actionContour.triggered.connect(self.apply_contour_detection)
        
        
        # Toolbar actions
        self.actionOpen.triggered.connect(self.load_image)
        self.actionSave.triggered.connect(self.save_image)
        self.actionReset.triggered.connect(self.reset_image)
        self.actionZoom_In.triggered.connect(self.apply_zoom_in_2x)
        self.actionZoom_Out.triggered.connect(self.apply_zoom_out_half)
        self.actionUndo.triggered.connect(self.undo_action)
        self.actionRedo.triggered.connect(self.redo_action)
        self.actionExit.triggered.connect(self.close)
        self.actionExport.triggered.connect(self.export_pixel_data)
    
    def load_image(self):
        """Load an image from file"""
        # Open file dialog to select an image file
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Image", 
            "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)", 
            options=options
        )
        
        # If user selected a file, load and display it
        if filePath:
            image = cv2.imread(filePath)
            if image is not None:
                self.processor.set_image(image)
                self.processor.original_image = image.copy()
                self.display_image(image, self.imglabel)
            else:
                QMessageBox.warning(self, "Error", "Failed to load image: " + filePath)
    
    def save_image(self):
        """Save the current image to file"""
        image = self.processor.get_image()
        if image is None:
            QMessageBox.warning(self, "Warning", "No image to save!")
            return
        
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Image", 
            "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)", 
            options=options
        )
        
        if filePath:
            # Add file extension if not provided
            if not any(filePath.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']):
                filePath += '.jpg'
                
            # Save the image
            try:
                cv2.imwrite(filePath, image)
                QMessageBox.information(self, "Success", f"Image saved successfully to:\n{filePath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")
    
    def reset_image(self):
        """Reset to the original image"""
        if hasattr(self.processor, 'original_image') and self.processor.original_image is not None:
            self.processor.image = self.processor.original_image.copy()
            self.display_image(self.processor.image, self.imglabelgray)
    
    def display_image(self, img, label):
        """Display an image on a QLabel"""
        if img is None:
            return
        
        # Determine the appropriate QImage format
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        
        # Create QImage and display it
        imgQ = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        self.imglabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.imglabelgray.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        imgQ = imgQ.rgbSwapped()
        label.setPixmap(QPixmap.fromImage(imgQ))
    
    # Basic operations
    def apply_grayscale(self):
        """Apply grayscale operation to the image"""
        image = self.processor.get_image()
        if image is None:
            return
        
        gray_image = grayscale(image)
        self.processor.set_image(gray_image)
        self.display_image(gray_image, self.imglabelgray)
    
    def apply_brightness(self):
        """Apply brightness adjustment to the image"""
        image = self.processor.get_image()
        if image is None:
            return
        
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            image = grayscale(image)
        
        bright_image = adjust_brightness(image, 80)
        self.processor.set_image(bright_image)
        self.display_image(bright_image, self.imglabelgray)
    
    def apply_contrast(self):
        """Apply contrast adjustment to the image"""
        image = self.processor.get_image()
        if image is None:
            return
        
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            image = grayscale(image)
        
        contrast_image = adjust_contrast(image, 1.7)
        self.processor.set_image(contrast_image)
        self.display_image(contrast_image, self.imglabelgray)
    
    def apply_contrast_stretching(self):
        """Apply contrast stretching to the image"""
        image = self.processor.get_image()
        if image is None:
            return
        
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            image = grayscale(image)
        
        stretched_image = contrast_stretching(image)
        self.processor.set_image(stretched_image)
        self.display_image(stretched_image, self.imglabelgray)
    
    def apply_negative(self):
        """Apply negative operation to the image"""
        image = self.processor.get_image()
        if image is None:
            return
        
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            image = grayscale(image)
        
        negative_image = negative(image)
        self.processor.set_image(negative_image)
        self.display_image(negative_image, self.imglabelgray)
    
    def apply_binarization(self):
        """Apply binarization to the image"""
        image = self.processor.get_image()
        if image is None:
            return
        
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            image = grayscale(image)
        
        binary_image = binarize(image, 180)
        self.processor.set_image(binary_image)
        self.display_image(binary_image, self.imglabelgray)
    
    # Placeholder methods for other operations that are connected but not implemented fully
    def show_gray_histogram(self):
        image = self.processor.get_image()
        if image is None:
            return
        if len(image.shape) == 3:
            image = grayscale(image)
        plot_grayscale_histogram(image)
        self.display_image(image, self.imglabelgray)
        plt.show()  # Menampilkan histogram
    
    def show_rgb_histogram(self):
        image = self.processor.get_image()
        if image is None:
            return
        plot_rgb_histogram(image)
        self.display_image(image, self.imglabelgray)
        plt.show()  # Menampilkan histogram
        
    def apply_histogram_equalization(self):
        image = self.processor.get_image()
        if image is None:
            return
        
        equalized_image, cdf_normalized, hist = calculate_equalized_histogram(image)
        
        plot_equalized_histogram(equalized_image)
        self.display_image(equalized_image, self.imglabelgray)
        plt.show()  # Menampilkan histogram
        
        
    def apply_translation(self):
        image = self.processor.get_image()
        if image is None:
            return
        translated_image = translate(image, dx=50, dy=50)  # Anda bisa ubah dx, dy sesuai kebutuhan
        self.processor.set_image(translated_image)
        self.display_image(translated_image, self.imglabelgray)

    def apply_rotate_90(self):
        image = self.processor.get_image()
        if image is None:
            return
        rotated_image = rotate90(image)
        self.processor.set_image(rotated_image)
        self.display_image(rotated_image, self.imglabelgray)

    def apply_rotate_45(self):
        image = self.processor.get_image()
        if image is None:
            return
        rotated_image = rotate(image, angle=45)
        self.processor.set_image(rotated_image)
        self.display_image(rotated_image, self.imglabelgray)

    def apply_rotate_neg_45(self):
        image = self.processor.get_image()
        if image is None:
            return
        rotated_image = rotate(image, angle=-45)
        self.processor.set_image(rotated_image)
        self.display_image(rotated_image, self.imglabelgray)

    def apply_rotate_180(self):
        image = self.processor.get_image()
        if image is None:
            return
        rotated_image = rotate180(image)
        self.processor.set_image(rotated_image)
        self.display_image(rotated_image, self.imglabelgray)

    def apply_rotate_neg_90(self):
        image = self.processor.get_image()
        if image is None:
            return
        rotated_image = rotate270(image)
        self.processor.set_image(rotated_image)
        self.display_image(rotated_image, self.imglabelgray)

    def apply_transpose(self):
        image = self.processor.get_image()
        if image is None:
            return
        transposed_image = transpose(image)
        self.processor.set_image(transposed_image)
        self.display_image(transposed_image, self.imglabelgray)

    def apply_zoom_in_2x(self):
        image = self.processor.get_image()
        if image is None:
            return
        zoomed_image = zoom_in(image, factor=2.0)
        self.processor.set_image(zoomed_image)
        self.display_image(zoomed_image, self.imglabelgray)

    def apply_zoom_in_3x(self):
        image = self.processor.get_image()
        if image is None:
            return
        zoomed_image = zoom_in(image, factor=3.0)
        self.processor.set_image(zoomed_image)
        self.display_image(zoomed_image, self.imglabelgray)

    def apply_zoom_in_4x(self):
        image = self.processor.get_image()
        if image is None:
            return
        zoomed_image = zoom_in(image, factor=4.0)
        self.processor.set_image(zoomed_image)
        self.display_image(zoomed_image, self.imglabelgray)

    def apply_zoom_out_half(self):
        image = self.processor.get_image()
        if image is None:
            return
        zoomed_image = zoom_out(image, factor=0.5)
        self.processor.set_image(zoomed_image)
        self.display_image(zoomed_image, self.imglabelgray)

    def apply_zoom_out_quarter(self):
        image = self.processor.get_image()
        if image is None:
            return
        zoomed_image = zoom_out(image, factor=0.25)
        self.processor.set_image(zoomed_image)
        self.display_image(zoomed_image, self.imglabelgray)

    def apply_zoom_out_three_quarters(self):
        image = self.processor.get_image()
        if image is None:
            return
        zoomed_image = zoom_out(image, factor=0.75)
        self.processor.set_image(zoomed_image)
        self.display_image(zoomed_image, self.imglabelgray)

    def apply_skew(self):
        image = self.processor.get_image()
        if image is None:
            return
        # Contoh: ubah lebar menjadi 1.5x dan tinggi menjadi 0.8x
        h, w = image.shape[:2]
        skewed_image = skew(image, new_width=int(w*1.5), new_height=int(h*0.8))
        self.processor.set_image(skewed_image)
        self.display_image(skewed_image, self.imglabelgray)

    def apply_crop(self):
        image = self.processor.get_image()
        if image is None:
            return
        # Contoh: crop 100x100 dari pojok kiri atas
        cropped_image = crop(image, x=0, y=0, width=100, height=100)
        self.processor.set_image(cropped_image)
        # Tampilkan hasil crop dengan imshow
        import matplotlib.pyplot as plt
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title('Hasil Crop')
        plt.axis('off')
        plt.show()
        # Tidak perlu display_image ke label
        
    def apply_add_subtract(self):
        """Menjalankan operasi penjumlahan dan pengurangan dua gambar contoh (kucing1.jpg dan kucing2.jpg)"""
        # Path gambar contoh
        img1_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'Example', 'kucing1.jpg')
        img2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'Example', 'kucing2.jpg')
        
        # Baca gambar langsung sebagai grayscale
        img1 = cv2.imread(img1_path, 0)  # 0 = grayscale
        img2 = cv2.imread(img2_path, 0)  # 0 = grayscale
        
        if img1 is None or img2 is None:
            QMessageBox.warning(self, "Error", "Gambar kucing1.jpg atau kucing2.jpg tidak ditemukan!")
            return
            
        # Resize agar sama jika perlu
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Operasi penjumlahan dan pengurangan menggunakan modul arithmetic
        image_tambah = add_images(img1, img2)
        image_kurang = subtract_images(img1, img2)
        
        # Tampilkan hasil dengan cv2.imshow
        cv2.imshow('image 1 original', img1)
        cv2.imshow('image 2 original', img2)
        cv2.imshow('image tambah', image_tambah)
        cv2.imshow('image kurang', image_kurang)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def apply_multiply_divide(self):
        """Menjalankan operasi perkalian dan pembagian dua gambar contoh (kucing1.jpg dan kucing2.jpg)"""
        # Path gambar contoh
        img1_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'Example', 'kucing1.jpg')
        img2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'Example', 'kucing2.jpg')
        
        # Baca gambar langsung sebagai grayscale
        img1 = cv2.imread(img1_path, 0)  # 0 = grayscale
        img2 = cv2.imread(img2_path, 0)  # 0 = grayscale
        
        if img1 is None or img2 is None:
            QMessageBox.warning(self, "Error", "Gambar kucing1.jpg atau kucing2.jpg tidak ditemukan!")
            return
            
        # Resize agar sama jika perlu
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Operasi perkalian dan pembagian menggunakan modul arithmetic
        image_kali = multiply_images(img1, img2)
        image_bagi = divide_images(img1, img2)
        
        # Tampilkan hasil dengan cv2.imshow
        cv2.imshow('image 1 original', img1)
        cv2.imshow('image 2 original', img2)
        cv2.imshow('image kali', image_kali)
        cv2.imshow('image bagi', image_bagi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def apply_bitwise_and(self):
        """Menjalankan operasi bitwise AND pada dua gambar contoh (kucing1.jpg dan kucing2.jpg)"""
        # Path gambar contoh
        img1_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'Example', 'kucing1.jpg')
        img2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'Example', 'kucing2.jpg')
        
        # Baca gambar RGB
        img1 = cv2.imread(img1_path, 1)  # 1 = color
        img2 = cv2.imread(img2_path, 1)  # 1 = color
        
        if img1 is None or img2 is None:
            QMessageBox.warning(self, "Error", "Gambar kucing1.jpg atau kucing2.jpg tidak ditemukan!")
            return
            
        # Resize agar sama jika perlu
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Konversi ke RGB untuk tampilan konsisten
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Operasi AND menggunakan modul arithmetic
        operasi = bitwise_and(img1_rgb, img2_rgb)
        
        # Tampilkan hasil dengan cv2.imshow
        cv2.imshow('image 1 original', cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow('image 2 original', cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow('image operasi AND', cv2.cvtColor(operasi, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_bitwise_or(self):
        """Menjalankan operasi bitwise OR pada dua gambar contoh (kucing1.jpg dan kucing2.jpg)"""
        # Path gambar contoh
        img1_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'Example', 'kucing1.jpg')
        img2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'Example', 'kucing2.jpg')
        
        # Baca gambar RGB
        img1 = cv2.imread(img1_path, 1)  # 1 = color
        img2 = cv2.imread(img2_path, 1)  # 1 = color
        
        if img1 is None or img2 is None:
            QMessageBox.warning(self, "Error", "Gambar kucing1.jpg atau kucing2.jpg tidak ditemukan!")
            return
            
        # Resize agar sama jika perlu
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Konversi ke RGB untuk tampilan konsisten
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Operasi OR menggunakan modul arithmetic
        operasi = bitwise_or(img1_rgb, img2_rgb)
        
        # Tampilkan hasil dengan cv2.imshow
        cv2.imshow('image 1 original', cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow('image 2 original', cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow('image operasi OR', cv2.cvtColor(operasi, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_bitwise_xor(self):
        """Menjalankan operasi bitwise XOR pada dua gambar contoh (kucing1.jpg dan kucing2.jpg)"""
        # Path gambar contoh
        img1_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'Example', 'kucing1.jpg')
        img2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'Example', 'kucing2.jpg')
        
        # Baca gambar RGB
        img1 = cv2.imread(img1_path, 1)  # 1 = color
        img2 = cv2.imread(img2_path, 1)  # 1 = color
        
        if img1 is None or img2 is None:
            QMessageBox.warning(self, "Error", "Gambar kucing1.jpg atau kucing2.jpg tidak ditemukan!")
            return
            
        # Resize agar sama jika perlu
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Konversi ke RGB untuk tampilan konsisten
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Operasi XOR menggunakan modul arithmetic
        operasi = bitwise_xor(img1_rgb, img2_rgb)
        
        # Tampilkan hasil dengan cv2.imshow
        cv2.imshow('image 1 original', cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow('image 2 original', cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow('image operasi XOR', cv2.cvtColor(operasi, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def apply_filtering(self):
        """Apply filtering (mean or Gaussian) to the image"""
        image = self.processor.get_image()
        if image is None:
            return
            
        # Create a dialog for filter selection
        dialog = QDialog(self)
        dialog.setWindowTitle("Filter Selection")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # Filter type selection
        filter_label = QLabel("Filter Type:")
        filter_combo = QComboBox()
        filter_combo.addItems(["Mean Filter", "Gaussian Filter"])
        
        # Filter size selection
        size_label = QLabel("Filter Size:")
        size_combo = QComboBox()
        size_combo.addItems(["3x3", "5x5", "7x7"])
        
        # Preserve color checkbox
        preserve_color_check = QCheckBox("Preserve Color (apply filter to each channel)")
        preserve_color_check.setChecked(len(image.shape) > 2)  # Default checked for color images
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(filter_label)
        layout.addWidget(filter_combo)
        layout.addWidget(size_label)
        layout.addWidget(size_combo)
        layout.addWidget(preserve_color_check)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            filter_type = filter_combo.currentText()
            filter_size = int(size_combo.currentText().split("x")[0])
            preserve_color = preserve_color_check.isChecked()
            
            # For color images when preserve_color is checked
            if len(image.shape) > 2 and preserve_color:
                # Split channels
                b, g, r = cv2.split(image)
                
                # Apply selected filter to each channel
                if filter_type == "Mean Filter":
                    b_filtered = mean_filter(b, filter_size)
                    g_filtered = mean_filter(g, filter_size)
                    r_filtered = mean_filter(r, filter_size)
                else:  # Gaussian Filter
                    if filter_size > 5:
                        # Since Gaussian filter only supports 3x3 and 5x5, use cv2 for larger sizes
                        b_filtered = cv2.GaussianBlur(b, (filter_size, filter_size), 0)
                        g_filtered = cv2.GaussianBlur(g, (filter_size, filter_size), 0)
                        r_filtered = cv2.GaussianBlur(r, (filter_size, filter_size), 0)
                    else:
                        b_filtered = gaussian_filter(b, filter_size)
                        g_filtered = gaussian_filter(g, filter_size)
                        r_filtered = gaussian_filter(r, filter_size)
                
                # Merge channels back
                filtered_image = cv2.merge([b_filtered, g_filtered, r_filtered])
            else:
                # Convert to grayscale if color image and not preserving color
                if len(image.shape) > 2 and not preserve_color:
                    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray_img = image.copy()
                
                # Apply selected filter
                if filter_type == "Mean Filter":
                    filtered_image = mean_filter(gray_img, filter_size)
                else:  # Gaussian Filter
                    if filter_size > 5:
                        # Since Gaussian filter only supports 3x3 and 5x5, use cv2 for larger sizes
                        filtered_image = cv2.GaussianBlur(gray_img, (filter_size, filter_size), 0)
                    else:
                        filtered_image = gaussian_filter(gray_img, filter_size)
            
            # Update and display the filtered image
            self.processor.set_image(filtered_image)
            self.display_image(filtered_image, self.imglabelgray)
        
    def apply_sharpening(self):
        """Apply sharpening filter to the image"""
        image = self.processor.get_image()
        if image is None:
            return
            
        # Create a dialog for sharpening parameter selection
        dialog = QDialog(self)
        dialog.setWindowTitle("Sharpening Options")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # Sharpening type selection
        type_label = QLabel("Sharpening Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Basic", "Strong", "Mild", "Laplacian", "High Boost"])
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(type_label)
        layout.addWidget(type_combo)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Get selected kernel type
            kernel_type_map = {
                "Basic": "basic",
                "Strong": "strong",
                "Mild": "mild",
                "Laplacian": "laplacian",
                "High Boost": "high_boost"
            }
            kernel_type = kernel_type_map[type_combo.currentText()]
            
            # Import the sharpen function
            from modules.filtering.sharpening import sharpen
            
            # Apply sharpening with selected kernel type
            sharpened_image = sharpen(image, kernel_type=kernel_type)
            
            # Update and display the sharpened image
            self.processor.set_image(sharpened_image)
            self.display_image(sharpened_image, self.imglabelgray)
        
    def apply_median_filter(self):
        """Apply median filter to the image"""
        image = self.processor.get_image()
        if image is None:
            return
            
        # Create a dialog for filter size selection
        dialog = QDialog(self)
        dialog.setWindowTitle("Median Filter Size")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # Filter size selection
        size_label = QLabel("Filter Size:")
        size_combo = QComboBox()
        size_combo.addItems(["3x3", "5x5", "7x7"])
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(size_label)
        layout.addWidget(size_combo)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            filter_size = int(size_combo.currentText().split("x")[0])
            
            # Apply median filter
            filtered_image = median_filter(image, filter_size)
            
            # Update and display the filtered image
            self.processor.set_image(filtered_image)
            self.display_image(filtered_image, self.imglabelgray)
        
    def apply_max_filter(self):
        """Apply max filter to the image"""
        image = self.processor.get_image()
        if image is None:
            return
            
        # Create a dialog for filter size selection
        dialog = QDialog(self)
        dialog.setWindowTitle("Max Filter Size")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # Filter size selection
        size_label = QLabel("Filter Size:")
        size_combo = QComboBox()
        size_combo.addItems(["3x3", "5x5", "7x7"])
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(size_label)
        layout.addWidget(size_combo)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            filter_size = int(size_combo.currentText().split("x")[0])
            
            # Apply max filter
            filtered_image = max_filter(image, filter_size)
            
            # Update and display the filtered image
            self.processor.set_image(filtered_image)
            self.display_image(filtered_image, self.imglabelgray)
        
    def apply_min_filter(self):
        """Apply min filter to the image"""
        image = self.processor.get_image()
        if image is None:
            return
            
        # Create a dialog for filter size selection
        dialog = QDialog(self)
        dialog.setWindowTitle("Min Filter Size")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # Filter size selection
        size_label = QLabel("Filter Size:")
        size_combo = QComboBox()
        size_combo.addItems(["3x3", "5x5", "7x7"])
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(size_label)
        layout.addWidget(size_combo)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            filter_size = int(size_combo.currentText().split("x")[0])
            
            # Apply min filter
            filtered_image = min_filter(image, filter_size)
            
            # Update and display the filtered image
            self.processor.set_image(filtered_image)
            self.display_image(filtered_image, self.imglabelgray)
        
    def apply_dft_smoothing(self):
        """Apply DFT-based smoothing filters to the image"""
        image = self.processor.get_image()
        if image is None:
            return
        
        # If image is not grayscale, convert it
        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
        
        # Create a dialog for filter selection
        dialog = QDialog(self)
        dialog.setWindowTitle("DFT Filter Selection")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # Filter type selection
        filter_label = QLabel("Filter Type:")
        filter_combo = QComboBox()
        filter_combo.addItems(["Low Pass Filter", "Band Pass Filter"])
        
        # Cutoff frequency selection for low pass filter
        cutoff_label = QLabel("Cutoff Radius (pixels):")
        cutoff_slider = QSlider(Qt.Horizontal)
        cutoff_slider.setMinimum(10)
        cutoff_slider.setMaximum(100)
        cutoff_slider.setValue(30)
        cutoff_value = QLabel("30 pixels")
        
        # Connect slider to label
        cutoff_slider.valueChanged.connect(lambda v: cutoff_value.setText(f"{v} pixels"))
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(filter_label)
        layout.addWidget(filter_combo)
        layout.addWidget(cutoff_label)
        layout.addWidget(cutoff_slider)
        layout.addWidget(cutoff_value)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            filter_type = filter_combo.currentText()
            cutoff_radius = cutoff_slider.value()
            
            # Import frequency domain functions
            from modules.filtering.frequency_domain import low_pass_filter, band_pass_filter, dft
            
            if filter_type == "Low Pass Filter":
                # Apply low-pass filter
                filtered_image, magnitude, filtered_magnitude = low_pass_filter(gray_image, radius=cutoff_radius)
                
                # Plot the magnitude spectrum before and after filtering
                plt.figure(figsize=(10, 5))
                plt.subplot(121), plt.imshow(magnitude, cmap='gray')
                plt.title('Original Magnitude Spectrum'), plt.axis('off')
                plt.subplot(122), plt.imshow(filtered_magnitude, cmap='gray')
                plt.title('Filtered Magnitude Spectrum'), plt.axis('off')
                plt.tight_layout()
                plt.show()
                
            else:  # Band Pass Filter
                # Apply band-pass filter
                inner_radius = cutoff_radius // 2
                outer_radius = cutoff_radius
                filtered_image, magnitude, filtered_magnitude = band_pass_filter(gray_image, 
                                                                            inner_radius=inner_radius, 
                                                                            outer_radius=outer_radius)
                
                # Plot the magnitude spectrum before and after filtering
                plt.figure(figsize=(10, 5))
                plt.subplot(121), plt.imshow(magnitude, cmap='gray')
                plt.title('Original Magnitude Spectrum'), plt.axis('off')
                plt.subplot(122), plt.imshow(filtered_magnitude, cmap='gray')
                plt.title('Filtered Magnitude Spectrum'), plt.axis('off')
                plt.tight_layout()
                plt.show()
            
            # Update and display the filtered image
            self.processor.set_image(filtered_image)
            self.display_image(filtered_image, self.imglabelgray)
        
    def apply_dft_edge_detection(self):
        """Apply DFT-based edge detection to the image"""
        image = self.processor.get_image()
        if image is None:
            return
        
        # If image is not grayscale, convert it
        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
        
        # Create a dialog for filter parameter selection
        dialog = QDialog(self)
        dialog.setWindowTitle("DFT Edge Detection Settings")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # Cutoff frequency selection
        cutoff_label = QLabel("Cutoff Radius (pixels):")
        cutoff_slider = QSlider(Qt.Horizontal)
        cutoff_slider.setMinimum(10)
        cutoff_slider.setMaximum(100)
        cutoff_slider.setValue(30)
        cutoff_value = QLabel("30 pixels")
        
        # Connect slider to label
        cutoff_slider.valueChanged.connect(lambda v: cutoff_value.setText(f"{v} pixels"))
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(cutoff_label)
        layout.addWidget(cutoff_slider)
        layout.addWidget(cutoff_value)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            cutoff_radius = cutoff_slider.value()
            
            # Import high-pass filter function
            from modules.filtering.frequency_domain import high_pass_filter
            
            # Apply high-pass filter for edge detection
            filtered_image, magnitude, filtered_magnitude = high_pass_filter(gray_image, radius=cutoff_radius)
            
            # Plot the magnitude spectrum before and after filtering
            plt.figure(figsize=(10, 5))
            plt.subplot(121), plt.imshow(magnitude, cmap='gray')
            plt.title('Original Magnitude Spectrum'), plt.axis('off')
            plt.subplot(122), plt.imshow(filtered_magnitude, cmap='gray')
            plt.title('High-Pass Filtered Spectrum'), plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            # Update and display the edge detection result
            self.processor.set_image(filtered_image)
            self.display_image(filtered_image, self.imglabelgray)
        
    def apply_sobel_edge_detection(self):
        """Apply Sobel edge detection to the image"""
        image = self.processor.get_image()
        if image is None:
            return
        
        # Import Sobel edge detection function
        from modules.edge_detection.detectors import sobel_edge_detection
        
        # Apply Sobel edge detection
        edge_image = sobel_edge_detection(image)
        
        # Update and display the edge detection result
        self.processor.set_image(edge_image)
        self.display_image(edge_image, self.imglabelgray)
        
    def apply_prewitt_edge_detection(self):
        """Apply Prewitt edge detection to the image"""
        image = self.processor.get_image()
        if image is None:
            return
        
        # Import Prewitt edge detection function
        from modules.edge_detection.detectors import prewitt_edge_detection
        
        # Apply Prewitt edge detection
        edge_image = prewitt_edge_detection(image)
        
        # Update and display the edge detection result
        self.processor.set_image(edge_image)
        self.display_image(edge_image, self.imglabelgray)
        
    def apply_robert_edge_detection(self):
        """Apply Roberts edge detection to the image"""
        image = self.processor.get_image()
        if image is None:
            return
        
        # Import Roberts edge detection function
        from modules.edge_detection.detectors import roberts_edge_detection
        
        # Apply Roberts edge detection
        edge_image = roberts_edge_detection(image)
        
        # Update and display the edge detection result
        self.processor.set_image(edge_image)
        self.display_image(edge_image, self.imglabelgray)
        
    def apply_canny_edge_detection(self):
        """Apply Canny edge detection to the image"""
        image = self.processor.get_image()
        if image is None:
            return
        
        # Create a dialog for Canny parameters
        dialog = QDialog(self)
        dialog.setWindowTitle("Canny Edge Detection Parameters")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # Lower threshold selection
        low_label = QLabel("Lower Threshold:")
        low_slider = QSlider(Qt.Horizontal)
        low_slider.setMinimum(0)
        low_slider.setMaximum(255)
        low_slider.setValue(15)
        low_value = QLabel("15")
        
        # Upper threshold selection
        high_label = QLabel("Upper Threshold:")
        high_slider = QSlider(Qt.Horizontal)
        high_slider.setMinimum(0)
        high_slider.setMaximum(255)
        high_slider.setValue(40)
        high_value = QLabel("40")
        
        # Connect sliders to labels
        low_slider.valueChanged.connect(lambda v: low_value.setText(str(v)))
        high_slider.valueChanged.connect(lambda v: high_value.setText(str(v)))
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(low_label)
        layout.addWidget(low_slider)
        layout.addWidget(low_value)
        layout.addWidget(high_label)
        layout.addWidget(high_slider)
        layout.addWidget(high_value)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            low_threshold = low_slider.value()
            high_threshold = high_slider.value()
            
            # Import Canny edge detection function
            from modules.edge_detection.detectors import canny_edge_detection
            
            # Apply Canny edge detection
            edge_image = canny_edge_detection(image, low_threshold, high_threshold)
            
            # Update and display the edge detection result
            self.processor.set_image(edge_image)
            self.display_image(edge_image, self.imglabelgray)
        
    def apply_dilation(self):
        """Apply dilation morphological operation to the image"""
        image = self.processor.get_image()
        if image is None:
            return
            
        # Create a dialog for dilation parameters
        dialog = QDialog(self)
        dialog.setWindowTitle("Dilation Parameters")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # Kernel size selection
        kernel_label = QLabel("Kernel Size:")
        kernel_combo = QComboBox()
        kernel_combo.addItems(["3x3", "5x5", "7x7", "9x9"])
        kernel_combo.setCurrentIndex(1)  # Default to 5x5
        
        # Iterations selection
        iterations_label = QLabel("Iterations:")
        iterations_spin = QSpinBox()
        iterations_spin.setMinimum(1)
        iterations_spin.setMaximum(10)
        iterations_spin.setValue(1)
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(kernel_label)
        layout.addWidget(kernel_combo)
        layout.addWidget(iterations_label)
        layout.addWidget(iterations_spin)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            kernel_size = int(kernel_combo.currentText().split("x")[0])
            iterations = iterations_spin.value()
            
            # Import dilation function
            from modules.morphology.operations import dilation
            
            # Apply dilation
            processed_image = dilation(image, kernel_size=kernel_size, iterations=iterations)
            
            # Update and display the result
            self.processor.set_image(processed_image)
            self.display_image(processed_image, self.imglabelgray)
        
    def apply_erosion(self):
        """Apply erosion morphological operation to the image"""
        image = self.processor.get_image()
        if image is None:
            return
            
        # Create a dialog for erosion parameters
        dialog = QDialog(self)
        dialog.setWindowTitle("Erosion Parameters")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # Kernel size selection
        kernel_label = QLabel("Kernel Size:")
        kernel_combo = QComboBox()
        kernel_combo.addItems(["3x3", "5x5", "7x7", "9x9"])
        kernel_combo.setCurrentIndex(1)  # Default to 5x5
        
        # Iterations selection
        iterations_label = QLabel("Iterations:")
        iterations_spin = QSpinBox()
        iterations_spin.setMinimum(1)
        iterations_spin.setMaximum(10)
        iterations_spin.setValue(1)
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(kernel_label)
        layout.addWidget(kernel_combo)
        layout.addWidget(iterations_label)
        layout.addWidget(iterations_spin)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            kernel_size = int(kernel_combo.currentText().split("x")[0])
            iterations = iterations_spin.value()
            
            # Import erosion function
            from modules.morphology.operations import erosion
            
            # Apply erosion
            processed_image = erosion(image, kernel_size=kernel_size, iterations=iterations)
            
            # Update and display the result
            self.processor.set_image(processed_image)
            self.display_image(processed_image, self.imglabelgray)
        
    def apply_opening(self):
        """Apply opening morphological operation (erosion followed by dilation) to the image"""
        image = self.processor.get_image()
        if image is None:
            return
            
        # Create a dialog for opening parameters
        dialog = QDialog(self)
        dialog.setWindowTitle("Opening Parameters")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # Kernel size selection
        kernel_label = QLabel("Kernel Size:")
        kernel_combo = QComboBox()
        kernel_combo.addItems(["3x3", "5x5", "7x7", "9x9"])
        kernel_combo.setCurrentIndex(1)  # Default to 5x5
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(kernel_label)
        layout.addWidget(kernel_combo)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            kernel_size = int(kernel_combo.currentText().split("x")[0])
            
            # Import opening function
            from modules.morphology.operations import opening
            
            # Apply opening
            processed_image = opening(image, kernel_size=kernel_size)
            
            # Update and display the result
            self.processor.set_image(processed_image)
            self.display_image(processed_image, self.imglabelgray)
        
    def apply_closing(self):
        """Apply closing morphological operation (dilation followed by erosion) to the image"""
        image = self.processor.get_image()
        if image is None:
            return
            
        # Create a dialog for closing parameters
        dialog = QDialog(self)
        dialog.setWindowTitle("Closing Parameters")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # Kernel size selection
        kernel_label = QLabel("Kernel Size:")
        kernel_combo = QComboBox()
        kernel_combo.addItems(["3x3", "5x5", "7x7", "9x9"])
        kernel_combo.setCurrentIndex(1)  # Default to 5x5
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(kernel_label)
        layout.addWidget(kernel_combo)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            kernel_size = int(kernel_combo.currentText().split("x")[0])
            
            # Import closing function
            from modules.morphology.operations import closing
            
            # Apply closing
            processed_image = closing(image, kernel_size=kernel_size)
            
            # Update and display the result
            self.processor.set_image(processed_image)
            self.display_image(processed_image, self.imglabelgray)
        
    def apply_skeletonizing(self):
        """Apply skeletonization to the image, reducing it to a skeleton 1-pixel wide"""
        image = self.processor.get_image()
        if image is None:
            return
        
        # Ask for confirmation since this operation can be time-consuming
        confirmation = QMessageBox.question(
            self, 
            "Skeletonization", 
            "Skeletonization can be time-consuming for large images. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirmation == QMessageBox.Yes:
            # Import skeletonize function
            from modules.morphology.operations import skeletonize
            
            # Display a progress dialog
            progress = QProgressDialog("Applying skeletonization...", None, 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            QApplication.processEvents()  # Ensure UI updates
            
            try:
                # Apply skeletonization
                processed_image = skeletonize(image)
                
                # Update and display the result
                self.processor.set_image(processed_image)
                self.display_image(processed_image, self.imglabelgray)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error during skeletonization: {str(e)}")
            finally:
                progress.close()
        
    def apply_binary_threshold(self):
        """Apply binary thresholding to the image"""
        image = self.processor.get_image()
        if image is None:
            return
        
        # Apply binary threshold
        thresholded_image = binary_threshold(image)
            
        # Update and display the result
        self.processor.set_image(thresholded_image)
        self.display_image(thresholded_image, self.imglabelgray)
        
    def apply_binary_inv_threshold(self):
        """Apply inverted binary thresholding to the image"""
        image = self.processor.get_image()
        if image is None:
            return
        
        # Apply binary inverse threshold
        thresholded_image = binary_inv_threshold(image)
            
        # Update and display the result
        self.processor.set_image(thresholded_image)
        self.display_image(thresholded_image, self.imglabelgray)
        
    def apply_trunc_threshold(self):
        """Apply truncated thresholding to the image"""
        image = self.processor.get_image()
        if image is None:
            return
        
        # Apply truncated threshold
        thresholded_image = trunc_threshold(image)
            
        # Update and display the result
        self.processor.set_image(thresholded_image)
        self.display_image(thresholded_image, self.imglabelgray)
        
    def apply_tozero_threshold(self):
        """Apply to-zero thresholding to the image"""
        image = self.processor.get_image()
        if image is None:
            return
                
        # Apply to-zero threshold
        thresholded_image = tozero_threshold(image)
            
        # Update and display the result
        self.processor.set_image(thresholded_image)
        self.display_image(thresholded_image, self.imglabelgray)
        
    def apply_tozero_inv_threshold(self):
        """Apply inverted to-zero thresholding to the image"""
        image = self.processor.get_image()
        if image is None:
            return
    
        # Apply inverted to-zero threshold
        thresholded_image = tozero_inv_threshold(image)
            
        # Update and display the result
        self.processor.set_image(thresholded_image)
        self.display_image(thresholded_image, self.imglabelgray)
        
    def apply_adaptive_mean_threshold(self):
        """Apply adaptive mean thresholding to the image"""
        image = self.processor.get_image()
        if image is None:
            return
            
        # Apply adaptive mean threshold
        thresholded_image = adaptive_mean_threshold(image)
            
        # Update and display the result
        self.processor.set_image(thresholded_image)
        self.display_image(thresholded_image, self.imglabelgray)
        
    def apply_adaptive_gaussian_threshold(self):
        """Apply adaptive Gaussian thresholding to the image"""
        image = self.processor.get_image()
        if image is None:
            return
                
        # Apply adaptive Gaussian threshold
        thresholded_image = adaptive_gaussian_threshold(image)
            
        # Update and display the result
        self.processor.set_image(thresholded_image)
        self.display_image(thresholded_image, self.imglabelgray)
        
    def apply_otsu_threshold(self):
        """Apply Otsu's thresholding to the image"""
        image = self.processor.get_image()
        if image is None:
            return
        
        # Apply Otsu threshold (doesn't need manual threshold selection as it's automatic)
        thresholded_image, threshold_value = otsu_threshold(image)
        
        # Show information about the selected threshold
        QMessageBox.information(
            self,
            "Otsu Threshold",
            f"Otsu algorithm automatically selected threshold value: {threshold_value:.1f}"
        )
        
        # Update and display the result
        self.processor.set_image(thresholded_image)
        self.display_image(thresholded_image, self.imglabelgray)
        
    def apply_contour_detection(self):
        """Apply contour detection to identify shapes in the image"""
        image = self.processor.get_image()
        if image is None:
            return
        
        # Create a dialog for contour detection options
        dialog = QDialog(self)
        dialog.setWindowTitle("Contour Detection Options")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # Option to show colored shapes
        color_shapes_check = QCheckBox("Color Detected Shapes")
        color_shapes_check.setChecked(True)
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(color_shapes_check)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            color_shapes = color_shapes_check.isChecked()
            
            # Import contour detection function
            from modules.segmentation.thresholding import contour_detection
            
            # Apply contour detection
            result_image, contours, shapes = contour_detection(image, color_shapes=color_shapes)
            
            # Show information about detected shapes
            shape_counts = {}
            for shape_info in shapes:
                shape_type = shape_info["shape"]
                if shape_type in shape_counts:
                    shape_counts[shape_type] += 1
                else:
                    shape_counts[shape_type] = 1
            
            shape_info_text = "Detected Shapes:\n"
            for shape_type, count in shape_counts.items():
                shape_info_text += f"{shape_type}: {count}\n"
            
            QMessageBox.information(self, "Shape Detection Results", shape_info_text)
            
            # Update and display the result
            self.processor.set_image(result_image)
            self.display_image(result_image, self.imglabelgray)
            
            from utils.helpers import compare_images
            # Compare original and processed images
            compare_images(image, result_image, "Original Image", "Contour Detection Result")
            plt.show()
    
    
    def undo_action(self):
        """Undo the last image processing action"""
        prev_image = self.processor.undo()
        if prev_image is not None:
            self.display_image(prev_image, self.imglabelgray)
        else:
            QMessageBox.information(self, "Undo", "Nothing to undo")
    
    def redo_action(self):
        """Redo the last undone image processing action"""
        next_image = self.processor.redo()
        if next_image is not None:
            self.display_image(next_image, self.imglabelgray)
        else:
            QMessageBox.information(self, "Redo", "Nothing to redo")
    
    def export_pixel_data(self):
        """Fungsi ekspor data piksel dengan pemilihan format melalui dialog."""
        image = self.processor.get_image()
        if image is None:
            QMessageBox.warning(self, "Peringatan", "Tidak ada gambar yang dimuat!")
            return

        # Dialog Pilihan Format
        dialog = QDialog(self)
        dialog.setWindowTitle("Pilih Format Ekspor")
        layout = QVBoxLayout(dialog)

        label = QLabel("Pilih format file:")
        layout.addWidget(label)

        format_combo = QComboBox()
        format_combo.addItems(["txt", "csv", "xlsx"])
        layout.addWidget(format_combo)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)

        dialog.setLayout(layout)

        if dialog.exec_() != QDialog.Accepted:
            return  # Jika user batal

        selected_format = format_combo.currentText()

        # Pilih Lokasi Simpan
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Simpan Data Piksel", "", f"*.{selected_format}"
        )

        if not file_path:
            return  # Jika user batal

        # Use the core function to export pixel data
        if self.processor.export_pixel_data(file_path, selected_format):
            QMessageBox.information(self, "Sukses", f"Data piksel berhasil diekspor ke {file_path}")
        else:
            QMessageBox.warning(self, "Gagal", "Gagal mengekspor data piksel!")

def run():
    app = QtWidgets.QApplication(sys.argv)
    window = ImageProcessingApp()
    window.setWindowTitle('RestoraPix')
    window.show()
    sys.exit(app.exec_())