import cv2
import numpy as np
import matplotlib.pyplot as plt

# 'image1.jpg' is not found, using 'V KOHLI.jpeg' which is available.
image = cv2.imread("V KOHLI.jpeg")

# Check if the image was loaded successfully
if image is None:
    print("Error: Image not loaded. Please ensure 'V KOHLI.jpeg' is uploaded to your Colab environment.")
else:
    kernel = np.array([[0, 1, 0],
                       [1, -8, 1],
                       [0, 1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)

    # Convert BGR to RGB for matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sharpened_rgb)
    plt.title('Sharpened Image')
    plt.axis('off')

    plt.show()
    cv2.imwrite('Sharpened_Image.jpg', sharpened)
    print("Image sharpened and saved as 'Sharpened_Image.jpg'")
