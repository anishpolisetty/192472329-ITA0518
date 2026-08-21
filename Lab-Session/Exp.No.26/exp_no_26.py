import cv2
import numpy as np # numpy is implicitly used by cv2, good practice to import
import matplotlib.pyplot as plt

# Define paths for the main image and the logo image
# 'V KOHLI.jpeg' is available in your Colab files
img_path = "V KOHLI.jpeg"
# For the logo, you need to upload 'logo_image.jpg' to your Colab environment
logo_path = "logo_image.jpg" 

try:
    # Read the main image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Main image not found at {img_path}. Please ensure it's uploaded.")

    # Read the logo image
    logo = cv2.imread(logo_path)
    if logo is None:
        # If logo is not found, create a dummy logo for demonstration or handle it
        print(f"Warning: Logo image not found at {logo_path}. Using a placeholder logo.")
        # Create a simple white rectangle as a placeholder logo
        logo = np.ones((50, 150, 3), dtype=np.uint8) * 255 # 50x150 white image
        cv2.putText(logo, "LOGO", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # Optionally, you might want to raise an error if a real logo is critical
        # raise FileNotFoundError(f"Logo image not found at {logo_path}. Please upload it.")

    h_logo, w_logo, _ = logo.shape
    h_img, w_img, _ = img.shape

    # Calculate position for the logo (centered)
    center_y = int(h_img / 2)
    center_x = int(w_img / 2)
    top_y = center_y - int(h_logo / 2)
    left_x = center_x - int(w_logo / 2)
    bottom_y = top_y + h_logo
    right_x = left_x + w_logo

    # Ensure logo fits within the image boundaries
    if top_y < 0: top_y = 0
    if left_x < 0: left_x = 0
    if bottom_y > h_img: bottom_y = h_img
    if right_x > w_img: right_x = w_img

    # Adjust logo size if it was clipped
    actual_h_logo = bottom_y - top_y
    actual_w_logo = right_x - left_x
    logo_resized = cv2.resize(logo, (actual_w_logo, actual_h_logo))

    # Extract the region of interest from the main image
    destination_roi = img[top_y:bottom_y, left_x:right_x]

    # Blend the logo with the destination region
    result = cv2.addWeighted(destination_roi, 1, logo_resized, 0.5, 0)

    # Place the result back into the main image
    img[top_y:bottom_y, left_x:right_x] = result

    # Convert BGR to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the watermarked image using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title('Watermarked Image')
    plt.axis('off')
    plt.show()

    # Save the watermarked image
    cv2.imwrite('watermarked_image.jpg', img)
    print("Watermarked image saved as 'watermarked_image.jpg'")

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
