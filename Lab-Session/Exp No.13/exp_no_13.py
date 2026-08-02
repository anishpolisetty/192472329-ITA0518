import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Assuming 'PICTURE.png' has been uploaded to /content/ as per previous interactions.
image_path_colab = "/content/PICTURE.png"

img = cv2.imread(image_path_colab)

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Could not load image from {image_path_colab}. Make sure the file exists and is accessible.")
else:
    rows, cols = img.shape[:2]

    # Using the same transformation points as previously established for perspective transformation
    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    dst_points = np.float32([[0, 0], [cols - 1, 0], [int(0.33 * cols), rows - 1], [int(0.66 * cols), rows - 1]])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    perspective_img = cv2.warpPerspective(img, M, (cols, rows))

    cv2.imwrite('Perspective_Transformed_Image.jpg', perspective_img)
    print(f"Image loaded successfully and perspective transformed image saved as 'Perspective_Transformed_Image.jpg'.")

    print("Displaying the Perspective Transformed Image:")
    cv2_imshow(perspective_img)
