# AIM:- Draw Rectangular shape and extract objects
# PROGRAM:-
import cv2
from google.colab.patches import cv2_imshow # Import Colab-compatible imshow

# Corrected path syntax and using an existing image file from Colab environment
img = cv2.imread("/content/V KOHLI.jpeg")
start_point = (50, 50)
end_point = (200, 200)
color = (0, 0, 255)
thickness = 2
rect_img = cv2.rectangle(img, start_point, end_point, color, thickness)
cv2_imshow(rect_img) # Use cv2_imshow for displaying in Colab

obj_img = img[start_point[1]:end_point[1], start_point[0]:end_point[0]]
cv2_imshow(obj_img) # Use cv2_imshow for displaying in Colab
cv2.imwrite('object.jpg', obj_img)

# cv2.waitKey(0) and cv2.destroyAllWindows() are not needed with cv2_imshow
