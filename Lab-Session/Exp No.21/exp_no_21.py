import cv2
import numpy as np
import matplotlib.pyplot as plt

# Convert BGR to RGB for matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sharpened_rgb)
plt.title('Sharpened Image')
plt.axis('off')

plt.show()

# Using 'V KOHLI.jpeg' as it's available in the Colab environment
img = cv2.imread("V KOHLI.jpeg")

# Check if the image was loaded successfully
if img is None:
    print("Error: Image not loaded. Please check the file path and ensure the image exists.")
else:
    kernel = np.array([[1,1,1], [1,-8,1], [1,1,1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    cv2.imwrite('Sharpened_Image.jpg', sharpened)
    print("Image sharpened and saved as 'Sharpened_Image.jpg'")
