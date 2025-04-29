import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load citra grayscale
image = cv2.imread('mata_katarak.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(6,6))
plt.title("Citra Asli")
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()

# 2. Pra-pengolahan: Crop dan Thresholding
cropped = image[100:1100, 100:1100]  # crop ke 1000x1000
_, binary = cv2.threshold(cropped, 100, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(6,6))
plt.title("Citra Biner Hasil Thresholding")
plt.imshow(binary, cmap='gray')
plt.axis('off')
plt.show()

# 3. Structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

# 4. Operasi Morphologi

# Closing: menutup lubang hitam kecil dalam objek putih
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Opening: menghilangkan noise putih kecil
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

plt.figure(figsize=(6,6))
plt.title("Hasil Segmentasi (Morphological Closing + Opening)")
plt.imshow(opened, cmap='gray')
plt.axis('off')
plt.show()


