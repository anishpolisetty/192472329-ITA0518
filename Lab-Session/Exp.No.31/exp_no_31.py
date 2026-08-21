import cv2
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib for displaying images

# Correct image path to one available in Colab environment
image_path = "V KOHLI.jpeg" # Or upload image1.jpg and use "image1.jpg"

try:
    img = cv2.imread(image_path)

    # Check if image was loaded successfully
    if img is None:
        raise FileNotFoundError(f"Error: Image not found at {image_path}. Please ensure it's uploaded to your Colab environment.")

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Convert BGR to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    opening_rgb = cv2.cvtColor(opening, cv2.COLOR_BGR2RGB)

    # Display the original and opened images using matplotlib
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(opening_rgb)
    plt.title('Opened Image')
    plt.axis('off')

    plt.show()

    # Save the opened image
    cv2.imwrite('Opened.jpg', opening)
    print("Opened image saved as 'Opened.jpg'")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# cv2.imshow and cv2.waitKey do not work in Google Colab
# cv2.destroyAllWindows() also not needed in Colab for display
