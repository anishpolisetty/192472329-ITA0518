# AIM:- Morphological operations using Top hat technique.
# PROGRAM:-
import cv2
import numpy as np # Import numpy for array operations
import matplotlib.pyplot as plt # Import matplotlib for displaying images

# Correct image path to one available in Colab environment
image_path = "V KOHLI.jpeg" # Using V KOHLI.jpeg as it is available in the kernel

try:
    input_image = cv2.imread(image_path) # Read the image in color

    # Check if image was loaded successfully
    if input_image is None:
        raise FileNotFoundError(f"Error: Image not found at {image_path}. Please ensure it's uploaded to your Colab environment.")

    filterSize = (3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)

    # Convert the input image to grayscale for morphological operations as Top Hat typically works on grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    tophat_img = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)

    # Display the original (color) and Top Hat images using matplotlib
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)) # Display original color image
    plt.title('Original Image (Color)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(tophat_img, cmap='gray') # Display Top Hat image in grayscale
    plt.title('Top Hat Image (Grayscale)')
    plt.axis('off')

    plt.show()

    # Save the Top Hat image
    cv2.imwrite("tophat.jpg", tophat_img)
    print("Top Hat image saved as 'tophat.jpg'")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# cv2.imshow and cv2.waitKey do not work in Google Colab
# cv2.destroyAllWindows() also not needed in Colab for display
