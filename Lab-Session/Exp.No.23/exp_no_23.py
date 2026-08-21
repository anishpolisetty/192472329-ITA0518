from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np

# Using 'V KOHLI.jpeg' as it is available in the Colab environment
image_path = "V KOHLI.jpeg"

try:
    im1 = Image.open(image_path)
    im2 = im1.filter(ImageFilter.UnsharpMask(radius=3, percent=200, threshold=5))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(im1)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(im2)
    plt.title('Sharpened Image (PIL)')
    plt.axis('off')

    plt.show()
    im2.save('Sharpened_Image_PIL.jpg')
    print("Image sharpened and saved as 'Sharpened_Image_PIL.jpg'")
except FileNotFoundError:
    print(f"Error: The file '{image_path}' was not found. Please ensure it's uploaded to your Colab environment.")
except Exception as e:
    print(f"An error occurred: {e}")
