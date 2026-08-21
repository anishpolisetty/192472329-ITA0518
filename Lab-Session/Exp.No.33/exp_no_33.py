# Morphological operations using Morphological Gradient technique.
# PROGRAM:-
import cv2
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib for displaying images

# Correct image path to one available in Colab environment
image_path = "V KOHLI.jpeg" # Using V KOHLI.jpeg as it is available in the kernel

try:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Load as grayscale for morphological gradient

    # Check if image was loaded successfully
    if img is None:
        raise FileNotFoundError(f"Error: Image not found at {image_path}. Please ensure it's uploaded to your Colab environment.")

    kernel = np.ones((3, 3), np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    # Display the original and gradient images using matplotlib
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray') # Display grayscale original image
    plt.title('Original Image (Grayscale)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gradient, cmap='gray') # Display grayscale gradient image
    plt.title('Morphological Gradient')
    plt.axis('off')

    plt.show()

    # Save the gradient image
    cv2.imwrite('Morphological_Gradient.jpg', gradient)
    print("Morphological gradient image saved as 'Morphological_Gradient.jpg'")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# cv2.imshow and cv2.waitKey do not work in Google Colab
# cv2.destroyAllWindows() also not needed in Colab for display
