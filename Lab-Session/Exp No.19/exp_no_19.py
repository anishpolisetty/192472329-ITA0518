import cv2
import numpy as np
from google.colab.patches import cv2_imshow # Import for displaying images

image_path_colab = "/content/PICTURE.png" # Use the Colab-accessible image path

img = cv2.imread(image_path_colab, 0) # Read in grayscale

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Could not load image from {image_path_colab}. Make sure the file exists and is accessible.")
else:
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    cv2.imwrite('Edge_detection.jpg', edges)
    print("Combined Sobel X and Y edge detection applied and edges saved as 'Edge_detection.jpg'.")
    print("Displaying the Combined Edges:")
    cv2_imshow(edges) # Display the edges
    print("Image was loaded successfully")
