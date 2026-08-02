import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load the saved affine transformed image
affine_transformed_display = cv2.imread("Affine_Transformed.jpg")

# Check if the image was loaded successfully
if affine_transformed_display is None:
    print("Error: Could not load 'Affine_Transformed.jpg'.")
else:
    print('Displaying Affine_Transformed.jpg:')
    cv2_imshow(affine_transformed_display)

# The provided path is a local Windows path and is not accessible in Google Colab.
# You need to use a path to an image uploaded to the Colab environment.
# Assuming 'PICTURE.png' has been uploaded as per previous interactions.
image_path_colab = "/content/PICTURE.png"

img = cv2.imread(image_path_colab)

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Could not load image from {image_path_colab}. Make sure the file exists and is accessible.")
else:
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, 1000], [0, 1, 500]])
    affine_img = cv2.warpAffine(img, M, (cols, rows))
    cv2.imwrite('Affine_Transformed.jpg', affine_img)
    print(f"Image loaded successfully and affine transformed image saved as 'Affine_Transformed.jpg'.")
    # If you want to display the image, use cv2_imshow from google.colab.patches
    # from google.colab.patches import cv2_imshow
    # cv2_imshow(affine_img)
