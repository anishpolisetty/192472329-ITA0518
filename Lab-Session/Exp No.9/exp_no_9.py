from google.colab.patches import cv2_imshow

# Load the saved image
rotated_image_display = cv2.imread("rotated_image.jpg")

# Display the rotated image
print('Displaying rotated_image.jpg:')
cv2_imshow(rotated_image_display)
