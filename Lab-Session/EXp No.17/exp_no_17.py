import cv2
import numpy as np
from google.colab.patches import cv2_imshow # Import for displaying images

image_path_colab = "/content/PICTURE.png" # Use the Colab-accessible image path

img = cv2.imread(image_path_colab, 0) # Read in grayscale

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Could not load image from {image_path_colab}. Make sure the file exists and is accessible.")
else:
    sobel_x = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)
    cv2.imwrite('sobel_x.jpg', sobel_x)
    print("Sobel X edge detection applied and edges saved as 'sobel_x.jpg'.")
    print("Displaying the Sobel X Edges:")
    cv2_imshow(sobel_x) # Display the edges
    print("Image was loaded successfully")
