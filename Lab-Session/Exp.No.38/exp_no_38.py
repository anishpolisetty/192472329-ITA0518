# AIM:- Face Detection
# PROGRAM:-
import cv2
import matplotlib.pyplot as plt
import os # Import os module to check for file existence

# Load the pre-trained Haar Cascade classifier for frontal face detection
# Ensure this XML file is available. It's usually part of OpenCV installations.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to the image file in Colab environment
image_path = "PICTUTE.png"

# Check if the image file exists. If not, use a known available image as a fallback.
if not os.path.exists(image_path):
    print(f"Warning: Image '{image_path}' not found. Please upload it to your Colab environment if you wish to use it.")
    print("Falling back to 'V KOHLI.jpeg' for demonstration.")
    image_path = "V KOHLI.jpeg" # Fallback to the known available image

try:
    # Read the image
    img = cv2.imread(image_path)

    # Check if image is loaded successfully (even after potential fallback)
    if img is None:
        raise FileNotFoundError(f"Error: Could not load image from {image_path}. It might be corrupted or still not exist.")

    # Convert the image to grayscale, as face detection works best on grayscale images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    # scaleFactor: Specifies how much the image size is reduced at each image scale.
    # minNeighbors: Specifies how many neighbors each candidate rectangle should have to retain it.
    # minSize: Minimum possible object size. Objects smaller than that are ignored.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green rectangle with thickness 2

    # Display the result using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for correct display
    plt.title(f'Face Detection on {os.path.basename(image_path)}')
    plt.axis('off') # Hide axes
    plt.show()

    if len(faces) > 0:
        print(f"Detected {len(faces)} face(s) in the image '{os.path.basename(image_path)}'.")
    else:
        print(f"No faces detected in the image '{os.path.basename(image_path)}'.")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Note: cv2.VideoCapture(0), cv2.imshow, cv2.waitKey, and cv2.destroyAllWindows
# are typically used for real-time video processing and display on a local machine
# and do not function in Google Colab environment directly. For displaying results,
# matplotlib.pyplot is used.
