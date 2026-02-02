import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("IMAGE.jpg", cv2.IMREAD_GRAYSCALE)

lpf_kernel = np.ones((3, 3), np.float32) / 9
low_pass = cv2.filter2D(img, -1, lpf_kernel)

hpf_kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
high_pass = cv2.filter2D(img, -1, hpf_kernel)

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Low Pass Filtered")
plt.imshow(low_pass, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("High Pass Filtered")
plt.imshow(high_pass, cmap="gray")
plt.axis("off")

plt.show()
