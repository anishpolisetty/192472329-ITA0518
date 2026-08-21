import cv2
import numpy as np
import matplotlib.pyplot as plt

def high_boost_filter(image, boost_factor):
    # Ensure image is float32 for calculations to avoid clipping
    image_float = image.astype(np.float32)
    kernel_size = 3
    # blur_factor is calculated but not used, so removing it.
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    blur_image = cv2.filter2D(image_float, -1, kernel)
    
    # High-boost filtering calculation
    mask = image_float + (image_float - blur_image) * boost_factor
    
    # Clip values to [0, 255] and convert back to uint8 for display/saving
    sharpened_image = np.clip(mask, 0, 255).astype(np.uint8)
    return sharpened_image

# Using 'V KOHLI.jpeg' as it is available in the Colab environment
image_path = "V KOHLI.jpeg"

try:
    # Read the image
    image = cv2.imread(image_path)

    # Check if image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Apply the high-boost filter
    sharpened_image = high_boost_filter(image, 1.5)

    # Convert BGR to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sharpened_image_rgb = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)

    # Display original and sharpened images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sharpened_image_rgb)
    plt.title('Sharpened Image (High-Boost)')
    plt.axis('off')

    plt.show()

    # Save the sharpened image
    cv2.imwrite('sharpened_image.jpg', sharpened_image)
    print("Image sharpened and saved as 'sharpened_image.jpg'")

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the image is uploaded to your Colab environment.")
except Exception as e:
    print(f"An error occurred: {e}")
