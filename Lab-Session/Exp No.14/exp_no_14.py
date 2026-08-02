import cv2
import numpy as np
from google.colab.patches import cv2_imshow # Import for displaying images

# Assuming 'PICTURE.png' has been uploaded to /content/ as per previous interactions.
image_path_colab = "/content/PICTURE.png"

img = cv2.imread(image_path_colab) # Use the Colab-accessible path

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Could not load image from {image_path_colab}. Make sure the file exists and is accessible.")
else:
    rows, cols = img.shape[:2]
    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    dst_points = np.float32([[0, 0], [cols - 1, 0], [0, int(0.7 * rows)], [cols - 1, int(0.7 * rows)]])
    M, _ = cv2.findHomography(src_points, dst_points)
    homography_img = cv2.warpPerspective(img, M, (cols, rows))
    cv2.imwrite('transformation_using_Homography_Image.jpg', homography_img)
    print(f"Homography transformation applied and image saved as 'transformation_using_Homography_Image.jpg'.")
    print("Displaying the Homography Transformed Image:")
    cv2_imshow(homography_img) # Display the transformed image
