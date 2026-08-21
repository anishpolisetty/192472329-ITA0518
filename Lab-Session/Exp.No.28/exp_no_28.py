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

    # Convert image to float32 for accurate sharpening, especially if negative values are generated
    img_float = img.astype(np.float32)

    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpened_img_float = cv2.filter2D(img_float, -1, kernel)

    # Clip values to [0, 255] and convert back to uint8 for display
    sharpened_img = np.clip(sharpened_img_float, 0, 255).astype(np.uint8)

    # Convert BGR to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sharpened_img_rgb = cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2RGB)

    # Display the input and sharpened images using matplotlib
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sharpened_img_rgb)
    plt.title('Sharpened Image')
    plt.axis('off')

    plt.show()

    # Save the sharpened image
    cv2.imwrite('sharpened_image.jpg', sharpened_img)
    print("Sharpened image saved as 'sharpened_image.jpg'")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# cv2.imshow and cv2.waitKey do not work in Google Colab
# cv2.destroyAllWindows() also not needed in Colab for display
