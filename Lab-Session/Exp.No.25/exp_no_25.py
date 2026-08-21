import cv2
import numpy as np
import matplotlib.pyplot as plt

# Using 'V KOHLI.jpeg' as it is available in the Colab environment
image_path = "V KOHLI.jpeg"

try:
    image = cv2.imread(image_path)

    # Check if image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}. Please ensure it's uploaded to your Colab environment.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to float for Sobel to handle negative gradients correctly
    gradient_x = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    gradient_y = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)

    # Calculate the magnitude of the gradient
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Normalize to 0-255 and convert to uint8 for display
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Display original and gradient images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title('Sobel Gradient Magnitude')
    plt.axis('off')

    plt.show()

    cv2.imwrite('sobel_gradient_image.jpg', gradient_magnitude)
    print("Sobel gradient image saved as 'sobel_gradient_image.jpg'")

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
