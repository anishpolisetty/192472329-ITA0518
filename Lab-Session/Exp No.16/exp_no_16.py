import cv2
import numpy as np
from google.colab.patches import cv2_imshow # Import for displaying images

image_path_colab = "/content/PICTURE.png" # Use the Colab-accessible image path

img = cv2.imread(image_path_colab, 0) # Read in grayscale for Canny

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Could not load image from {image_path_colab}. Make sure the file exists and is accessible.")
else:
    edges = cv2.Canny(img, 100, 200)
    cv2.imwrite('Edges.jpg', edges)
    print("Canny edge detection applied and edges saved as 'Edges.jpg'.")
    print("Displaying the Edges:")
    cv2_imshow(edges) # Display the edges
    print("Image loaded successfully")
