import cv2
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib for displaying images

# Correct image path to one available in Colab environment
image_path = "V KOHLI.jpeg" # Using V KOHLI.jpeg as it is available in the kernel

try:
    img = cv2.imread(image_path) # Read the image in color (BGR format)

    # Check if image was loaded successfully
    if img is None:
        raise FileNotFoundError(f"Error: Image not found at {image_path}. Please ensure it's uploaded to your Colab environment.")

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) # Apply closing on the color image

    # Convert original BGR to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert closed image to grayscale for display
    closing_gray = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)

    # Display the original and closed images using matplotlib
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb) # Display color original image
    plt.title('Original Image (Color)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(closing_gray, cmap='gray') # Display grayscale closed image
    plt.title('Closed Image (Grayscale Output)')
    plt.axis('off')

    plt.show()

    # Save the closed image (will save in BGR as it was processed in BGR)
    cv2.imwrite('Closing.jpg', closing)
    print("Closed image saved as 'Closing.jpg'")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# cv2.imshow and cv2.waitKey do not work in Google Colab
# cv2.destroyAllWindows() also not needed in Colab for display
