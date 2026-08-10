# ==============================================================
# 10. INTEGRATED FEATURE DETECTION PIPELINE
# Preprocessing + Edge Detection + SIFT Feature Extraction
# ==============================================================

# --------------------------------------------------------------
# STEP 1: IMPORT LIBRARIES
# --------------------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from google.colab import files


# ==============================================================
# STEP 2: UPLOAD INPUT IMAGE
# ==============================================================

print("Upload an image for the experiment:")

uploaded = files.upload()

image_path = list(uploaded.keys())[0]

print("Image uploaded:", image_path)


# ==============================================================
# STEP 3: LOAD ORIGINAL IMAGE
# ==============================================================

original = cv2.imread(image_path)

if original is None:
    raise Exception("Unable to read the image.")

# Resize image
original = cv2.resize(
    original,
    (800, 600)
)

print("\nImage loaded successfully.")
print("Image size:", original.shape)


# ==============================================================
# STEP 4: DISPLAY ORIGINAL IMAGE
# ==============================================================

plt.figure(figsize=(8, 6))

plt.imshow(
    cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Original Input Image")

plt.axis("off")

plt.show()


# ==============================================================
# STEP 5: PREPROCESSING
# ==============================================================
# Preprocessing includes:
# 1. Grayscale conversion
# 2. Gaussian Blur
# 3. Histogram Equalization
# ==============================================================

start_preprocessing = time.perf_counter()


# Convert to grayscale
gray = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2GRAY
)


# Gaussian Blur
blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# Histogram Equalization
preprocessed = cv2.equalizeHist(
    blurred
)


preprocessing_time = (
    time.perf_counter()
    - start_preprocessing
)


# ==============================================================
# STEP 6: DISPLAY PREPROCESSING STAGES
# ==============================================================

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)

plt.imshow(
    gray,
    cmap="gray"
)

plt.title("Grayscale")

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    blurred,
    cmap="gray"
)

plt.title("Gaussian Blur")

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    preprocessed,
    cmap="gray"
)

plt.title("Histogram Equalization")

plt.axis("off")


plt.tight_layout()

plt.show()


# ==============================================================
# STEP 7: EDGE DETECTION USING CANNY
# ==============================================================

start_edge = time.perf_counter()


edges = cv2.Canny(
    preprocessed,
    threshold1=100,
    threshold2=200
)


edge_time = (
    time.perf_counter()
    - start_edge
)


# Count edge pixels
edge_pixel_count = np.count_nonzero(
    edges
)


# ==============================================================
# STEP 8: DISPLAY EDGE IMAGE
# ==============================================================

plt.figure(figsize=(8, 6))

plt.imshow(
    edges,
    cmap="gray"
)

plt.title(
    "Canny Edge Detection"
)

plt.axis("off")

plt.show()


# ==============================================================
# STEP 9: CREATE SIFT FEATURE DETECTOR
# ==============================================================

sift = cv2.SIFT_create(
    nfeatures=1500
)


# ==============================================================
# STEP 10: SIFT FEATURE EXTRACTION
# ==============================================================

start_sift = time.perf_counter()


keypoints, descriptors = (
    sift.detectAndCompute(
        preprocessed,
        None
    )
)


sift_time = (
    time.perf_counter()
    - start_sift
)


# Number of keypoints
keypoint_count = len(keypoints)


# ==============================================================
# STEP 11: CALCULATE FEATURE DENSITY
# ==============================================================

height, width = preprocessed.shape

total_pixels = height * width

feature_density = (
    keypoint_count /
    total_pixels
) * 1000


# ==============================================================
# STEP 12: DRAW SIFT KEYPOINTS
# ==============================================================

keypoint_image = cv2.drawKeypoints(
    original,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)


# ==============================================================
# STEP 13: DISPLAY SIFT FEATURES
# ==============================================================

plt.figure(figsize=(10, 7))

plt.imshow(
    cv2.cvtColor(
        keypoint_image,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "SIFT Feature / Keypoint Detection"
)

plt.axis("off")

plt.show()


# ==============================================================
# STEP 14: DISPLAY COMPLETE PIPELINE
# ==============================================================

plt.figure(figsize=(16, 10))


plt.subplot(2, 2, 1)

plt.imshow(
    cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "1. Original Image"
)

plt.axis("off")


plt.subplot(2, 2, 2)

plt.imshow(
    preprocessed,
    cmap="gray"
)

plt.title(
    "2. Preprocessed Image"
)

plt.axis("off")


plt.subplot(2, 2, 3)

plt.imshow(
    edges,
    cmap="gray"
)

plt.title(
    "3. Canny Edge Detection"
)

plt.axis("off")


plt.subplot(2, 2, 4)

plt.imshow(
    cv2.cvtColor(
        keypoint_image,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "4. SIFT Feature Extraction"
)

plt.axis("off")


plt.tight_layout()

plt.show()


# ==============================================================
# STEP 15: PRINT QUANTITATIVE RESULTS
# ==============================================================

print("\n")
print("=" * 65)
print("          INTEGRATED FEATURE DETECTION RESULTS")
print("=" * 65)


print(
    "\nImage Dimensions:",
    width,
    "x",
    height
)

print(
    "Total Pixels:",
    total_pixels
)

print(
    "Number of Edge Pixels:",
    edge_pixel_count
)

print(
    "Number of SIFT Keypoints:",
    keypoint_count
)

print(
    "Feature Density:",
    round(
        feature_density,
        4
    ),
    "features / 1000 pixels"
)

print(
    "\nPreprocessing Time:",
    round(
        preprocessing_time * 1000,
        3
    ),
    "ms"
)

print(
    "Edge Detection Time:",
    round(
        edge_time * 1000,
        3
    ),
    "ms"
)

print(
    "SIFT Extraction Time:",
    round(
        sift_time * 1000,
        3
    ),
    "ms"
)


# ==============================================================
# STEP 16: TOTAL PROCESSING TIME
# ==============================================================

total_time = (
    preprocessing_time
    + edge_time
    + sift_time
)


print(
    "Total Pipeline Time:",
    round(
        total_time * 1000,
        3
    ),
    "ms"
)


# ==============================================================
# STEP 17: CREATE RESULTS TABLE
# ==============================================================

results = pd.DataFrame({

    "Pipeline Stage": [
        "Preprocessing",
        "Edge Detection",
        "Feature Extraction",
        "Complete Pipeline"
    ],

    "Output": [
        "Enhanced grayscale image",
        str(edge_pixel_count)
        + " edge pixels",
        str(keypoint_count)
        + " SIFT keypoints",
        "Final feature detection"
    ],

    "Processing Time (ms)": [

        round(
            preprocessing_time * 1000,
            3
        ),

        round(
            edge_time * 1000,
            3
        ),

        round(
            sift_time * 1000,
            3
        ),

        round(
            total_time * 1000,
            3
        )
    ]
})


print("\n")
print("=" * 65)
print("                 RESULTS TABLE")
print("=" * 65)

print(
    results.to_string(
        index=False
    )
)


# ==============================================================
# STEP 18: PROCESSING TIME GRAPH
# ==============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results["Pipeline Stage"],
    results["Processing Time (ms)"]
)

plt.xlabel(
    "Pipeline Stage"
)

plt.ylabel(
    "Processing Time (ms)"
)

plt.title(
    "Processing Time of Integrated Pipeline"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.show()


# ==============================================================
# STEP 19: EDGE AND FEATURE COUNT GRAPH
# ==============================================================

plt.figure(figsize=(8, 6))

plt.bar(
    [
        "Edge Pixels",
        "SIFT Keypoints"
    ],
    [
        edge_pixel_count,
        keypoint_count
    ]
)

plt.xlabel(
    "Detected Feature Type"
)

plt.ylabel(
    "Number of Features"
)

plt.title(
    "Edge and Keypoint Detection Analysis"
)

plt.tight_layout()

plt.show()


# ==============================================================
# STEP 20: FEATURE DENSITY VISUALIZATION
# ==============================================================

plt.figure(figsize=(7, 6))

plt.bar(
    ["SIFT Feature Density"],
    [feature_density]
)

plt.ylabel(
    "Features per 1000 Pixels"
)

plt.title(
    "SIFT Feature Density"
)

plt.tight_layout()

plt.show()


# ==============================================================
# STEP 21: SAVE INTERMEDIATE AND FINAL OUTPUTS
# ==============================================================

cv2.imwrite(
    "01_original_image.jpg",
    original
)

cv2.imwrite(
    "02_preprocessed_image.jpg",
    preprocessed
)

cv2.imwrite(
    "03_edge_detection.jpg",
    edges
)

cv2.imwrite(
    "04_sift_features.jpg",
    keypoint_image
)

results.to_csv(
    "pipeline_results.csv",
    index=False
)


print("\n")
print("=" * 65)
print("                 FILES SAVED")
print("=" * 65)

print("01_original_image.jpg")
print("02_preprocessed_image.jpg")
print("03_edge_detection.jpg")
print("04_sift_features.jpg")
print("pipeline_results.csv")


# ==============================================================
# STEP 22: FINAL OBSERVATIONS
# ==============================================================

print("\n")
print("=" * 65)
print("                    OBSERVATIONS")
print("=" * 65)

print("""
1. PREPROCESSING

2. EDGE DETECTION

3. FEATURE EXTRACTION

4. PERFORMANCE
  
5. INTEGRATED PIPELINE
  
""")


# ==============================================================
# STEP 23: FINAL CONCLUSION
# ==============================================================

print("\n")
print("=" * 65)
print("                    CONCLUSION")
print("=" * 65)

print("""
The integrated image processing pipeline successfully
combined preprocessing, Canny edge detection and SIFT
feature extraction.
""")
