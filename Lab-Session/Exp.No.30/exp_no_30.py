import cv2
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib for displaying images

# Correct image path to one available in Colab environment
image_path = "V KOHLI.jpeg" # Or upload image1.jpg and use "image1.jpg"

try:
    img = cv2.imread(image_path)

    # Check if image was loaded successfully
    if img is None:
        raise FileNotFoundError(f"Error: Image not found at {image_path}. Please ensure it's uploaded to your Colab environment.")

    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)

    # Convert BGR to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dilation_rgb = cv2.cvtColor(dilation, cv2.COLOR_BGR2RGB)

    # Display the original and dilated images using matplotlib
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(dilation_rgb)
    plt.title('Dilated Image')
    plt.axis('off')

    plt.show()

    # Save the dilated image
    cv2.imwrite('dilated_image.jpg', dilation)
    print("Dilated image saved as 'dilated_image.jpg'")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# cv2.imshow and cv2.waitKey do not work in Google Colab
# cv2.destroyAllWindows() also not needed in Colab for display
