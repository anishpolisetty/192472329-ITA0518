import cv2
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib for displaying images

# Correct image path to one available in Colab environment
image_path = "V KOHLI.jpeg" # Or upload image1.jpg and use "image1.jpg"

try:
    img = cv2.imread(image_path) # Read as a color image

    # Check if image was loaded successfully
    if img is None:
        raise FileNotFoundError(f"Error: Image not found at {image_path}. Please ensure it's uploaded to your Colab environment.")

    kernel = np.ones((5, 5), np.uint8)
    eroded_img = cv2.erode(img, kernel, iterations=1)

    # Convert BGR to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    eroded_img_rgb = cv2.cvtColor(eroded_img, cv2.COLOR_BGR2RGB)

    # Display the original and eroded images using matplotlib
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb) # Display original as color image
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(eroded_img_rgb) # Display eroded as color image
    plt.title('Eroded Image')
    plt.axis('off')

    plt.show()

    # Save the eroded image
    cv2.imwrite('eroded_image.jpg', eroded_img)
    print("Eroded image saved as 'eroded_image.jpg'")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# cv2.imshow and cv2.waitKey do not work in Google Colab
# cv2.destroyAllWindows() also not needed in Colab for display
