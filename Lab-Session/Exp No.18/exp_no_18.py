import cv2
import numpy as np
from google.colab.patches import cv2_imshow # Import for displaying images

image_path_colab = "/content/PICTURE.png" # Use the Colab-accessible image path

img = cv2.imread(image_path_colab, 0) # Read in grayscale

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Could not load image from {image_path_colab}. Make sure the file exists and is accessible.")
else:
    sobel_y = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=5)
    cv2.imwrite('sobel_y.jpg', sobel_y)
    print("Sobel Y edge detection applied and edges saved as 'sobel_y.jpg'.")
    print("Displaying the Sobel Y Edges:")
    cv2_imshow(sobel_y) # Display the edges
    print("Image was loaded successfully")
