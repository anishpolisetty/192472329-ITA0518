import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Read the perspective-transformed image
perspective_transformed_display = cv2.imread('Perspective_Transformed_Image.jpg')

# Check if the image was loaded successfully
if perspective_transformed_display is None:
    print("Error: Could not load 'Perspective_Transformed_Image.jpg'.")
else:
    print("Displaying the Perspective Transformed Image:")
    cv2_imshow(perspective_transformed_display)

# The previous path was a local Windows path and had an unmatched parenthesis.
# Correcting the syntax and updating to a Colab-accessible path.
# Assuming 'PICTURE.png' has been uploaded as per previous interactions.
image_path_colab = "/content/PICTURE.png"

img = cv2.imread(image_path_colab)

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Could not load image from {image_path_colab}. Make sure the file exists and is accessible.")
else:
    rows, cols = img.shape[:2]
    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    dst_points = np.float32([[0, 0], [cols - 1, 0], [int(0.33*cols), rows - 1], [int(0.66*cols), rows - 1]])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    perspective_img = cv2.warpPerspective(img, M, (cols, rows))
    cv2.imwrite('Perspective_Transformed_Image.jpg', perspective_img)
    print(f"Image loaded successfully and perspective transformed image saved as 'Perspective_Transformed_Image.jpg'.")
    # If you want to display the image, you would use cv2_imshow from google.colab.patches
    # from google.colab.patches import cv2_imshow
    # cv2_imshow(perspective_img)
