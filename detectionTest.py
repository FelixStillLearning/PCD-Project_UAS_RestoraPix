import cv2
from ultralytics import YOLO

# Pilih mode: "image" untuk gambar, "video" untuk kamera, "file" untuk video
mode = "video"  # Ubah ke "image" atau "file" sesuai kebutuhan

# Load model YOLOv5
model = YOLO("yolov5s.pt")  # Gunakan model YOLOv5 kecil (yolov5s)

if mode == "image":
    # Path gambar
    image_path = "kucing1.jpg"  # Ganti dengan path gambar yang ingin dideteksi
    image = cv2.imread(image_path)

    # Deteksi objek dalam gambar
    results = model(image_path)

    # Loop melalui hasil deteksi
    for result in results:
        # Konversi hasil ke gambar dengan bounding box
        img_with_boxes = result.plot()

        # Tampilkan gambar dengan OpenCV
        cv2.imshow("Deteksi Objek", img_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

elif mode == "video":
    # Buka kamera laptop
    cap = cv2.VideoCapture(0)  # Gunakan kamera default

elif mode == "file":
    # Buka video dari file
    cap = cv2.VideoCapture("video.mp4")  # Ganti dengan path video yang ingin dideteksi

else:
    print("Mode tidak valid! Pilih 'image', 'video', atau 'file'.")
    exit()

# Loop untuk membaca frame dari kamera atau video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek dalam frame
    results = model(frame)

    # Loop melalui hasil deteksi
    for result in results:
        # Konversi hasil ke gambar dengan bounding box
        img_with_boxes = result.plot()

        # Tampilkan hasil deteksi
        cv2.imshow("Deteksi Objek", img_with_boxes)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()