import cv2
from google.colab.patches import cv2_imshow

# In Python, backslashes in strings are interpreted as escape sequences.
# For file paths, especially on Windows, this can lead to errors like 'unicodeescape'
# if an invalid escape sequence is encountered (e.g., '\Users' where '\U' expects 8 hex digits).
# To prevent this, use raw strings by prefixing the string with 'r',
# or use forward slashes (/) which are also understood by Python on Windows.
#
# Also, note that C:\ drive paths are not accessible in Google Colab.
# You need to use paths to files uploaded to the Colab environment,
# for example, '/content/PICTURE.png' if you've uploaded a file with that name.

# Attempting to load 'image1.jpg' from a local Windows path will fail in Colab.
# Let's use the previously uploaded 'PICTURE.png' as a general input image.
image_path_colab = "/content/PICTURE.png"

image = cv2.imread(image_path_colab)

if image is None:
    print(f"Error: Could not load image from {image_path_colab}. Make sure the file exists and is accessible.")
else:
    # Get dimensions (if image loaded successfully)
    width = image.shape[1]
    height = image.shape[0]
    print(f"Image loaded successfully. Dimensions: {width}x{height}")

    # For displaying images in Google Colab, cv2.imshow is not supported.
    # We use cv2_imshow from google.colab.patches instead.
    # Window management functions like cv2.getWindowProperty, cv2.moveWindow,
    # cv2.waitKey, and cv2.destroyAllWindows are also not applicable in Colab.
    print('Displaying image:')
    cv2_imshow(image)
