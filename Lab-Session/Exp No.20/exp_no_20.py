import cv2
import numpy as np
from google.colab.patches import cv2_imshow # Import for displaying images

image_path_colab = "/content/PICTURE.png" # Use the Colab-accessible image path

img = cv2.imread(image_path_colab) # Read the image

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Could not load image from {image_path_colab}. Make sure the file exists and is accessible.")
else:
    kernel = np.array([[0,1,0], [1,-4,1], [0,1,0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    cv2.imwrite('Sharpened_Image.jpg', sharpened)
    print("Image sharpened and saved as 'Sharpened_Image.jpg'.")
    print("Displaying the Sharpened Image:")
    cv2_imshow(sharpened) # Display the sharpened image
    print("Image was loaded successfully")
