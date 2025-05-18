import cv2
import matplotlib.pyplot as plt

# Load gambar kucing (ganti path jika perlu)
img = cv2.imread('kucing1.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny edge detection OpenCV
edges_cv = cv2.Canny(gray_img, 100, 200)

# Tampilkan hasil
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Input (Grayscale)')
plt.imshow(gray_img, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Canny OpenCV (100,200)')
plt.imshow(edges_cv, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()