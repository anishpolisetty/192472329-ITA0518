# ==============================================================
# 9. PREPROCESSING VS FEATURE EXTRACTION
# Comparative Study using SIFT
# ==============================================================

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from google.colab import files


# ==============================================================
# STEP 1: UPLOAD TWO IMAGES
# ==============================================================

print("Upload Image 1 - Reference Image")
uploaded1 = files.upload()

print("Upload Image 2 - Test Image")
uploaded2 = files.upload()


# Get uploaded filenames automatically
image1_path = list(uploaded1.keys())[0]
image2_path = list(uploaded2.keys())[0]


# ==============================================================
# STEP 2: READ IMAGES
# ==============================================================

img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

if img1 is None or img2 is None:
    raise Exception("Unable to read the uploaded images.")


# Resize images to the same size
img1 = cv2.resize(img1, (800, 600))
img2 = cv2.resize(img2, (800, 600))


# ==============================================================
# STEP 3: CONVERT TO GRAYSCALE
# ==============================================================

gray1 = cv2.cvtColor(
    img1,
    cv2.COLOR_BGR2GRAY
)

gray2 = cv2.cvtColor(
    img2,
    cv2.COLOR_BGR2GRAY
)


# ==============================================================
# STEP 4: DISPLAY ORIGINAL IMAGES
# ==============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.imshow(
    cv2.cvtColor(
        img1,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Image 1 - Reference")
plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    cv2.cvtColor(
        img2,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Image 2 - Test")
plt.axis("off")

plt.tight_layout()
plt.show()


# ==============================================================
# STEP 5: PREPROCESSING FUNCTION
# ==============================================================

def preprocess_image(gray):

    # Gaussian Blur for noise reduction
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        blurred
    )

    return enhanced


# ==============================================================
# STEP 6: CREATE PREPROCESSED IMAGES
# ==============================================================

processed1 = preprocess_image(
    gray1
)

processed2 = preprocess_image(
    gray2
)


# ==============================================================
# STEP 7: DISPLAY PREPROCESSING
# ==============================================================

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)

plt.imshow(
    gray1,
    cmap="gray"
)

plt.title(
    "Image 1 - Without Preprocessing"
)

plt.axis("off")


plt.subplot(2, 2, 2)

plt.imshow(
    processed1,
    cmap="gray"
)

plt.title(
    "Image 1 - With Preprocessing"
)

plt.axis("off")


plt.subplot(2, 2, 3)

plt.imshow(
    gray2,
    cmap="gray"
)

plt.title(
    "Image 2 - Without Preprocessing"
)

plt.axis("off")


plt.subplot(2, 2, 4)

plt.imshow(
    processed2,
    cmap="gray"
)

plt.title(
    "Image 2 - With Preprocessing"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ==============================================================
# STEP 8: CREATE SIFT DETECTOR
# ==============================================================

sift = cv2.SIFT_create(
    nfeatures=1500
)


# ==============================================================
# STEP 9: CREATE FLANN MATCHER
# ==============================================================

FLANN_INDEX_KDTREE = 1

index_params = {
    "algorithm": FLANN_INDEX_KDTREE,
    "trees": 5
}

search_params = {
    "checks": 50
}

flann = cv2.FlannBasedMatcher(
    index_params,
    search_params
)


# ==============================================================
# STEP 10: FUNCTION FOR FEATURE EXTRACTION AND MATCHING
# ==============================================================

def perform_experiment(
    image1,
    image2,
    condition
):

    start_time = time.perf_counter()


    # ----------------------------------------------------------
    # SIFT FEATURE EXTRACTION
    # ----------------------------------------------------------

    keypoints1, descriptors1 = (
        sift.detectAndCompute(
            image1,
            None
        )
    )

    keypoints2, descriptors2 = (
        sift.detectAndCompute(
            image2,
            None
        )
    )


    # Check descriptors
    if descriptors1 is None or descriptors2 is None:

        return {
            "Condition": condition,
            "Keypoints Image 1": len(keypoints1),
            "Keypoints Image 2": len(keypoints2),
            "Good Matches": 0,
            "Match Percentage": 0,
            "Inliers": 0,
            "Inlier Percentage": 0,
            "Processing Time (ms)": 0,
            "Keypoints1": keypoints1,
            "Keypoints2": keypoints2,
            "Good": []
        }


    # ----------------------------------------------------------
    # FEATURE MATCHING
    # ----------------------------------------------------------

    matches = flann.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )


    # ----------------------------------------------------------
    # LOWE'S RATIO TEST
    # ----------------------------------------------------------

    good_matches = []

    for pair in matches:

        if len(pair) == 2:

            m, n = pair

            if m.distance < 0.75 * n.distance:

                good_matches.append(m)


    # ----------------------------------------------------------
    # MATCH PERCENTAGE
    # ----------------------------------------------------------

    match_percentage = (
        len(good_matches)
        / len(matches)
        * 100
        if len(matches) > 0
        else 0
    )


    # ----------------------------------------------------------
    # RANSAC INLIER ANALYSIS
    # ----------------------------------------------------------

    inliers = 0

    if len(good_matches) >= 4:

        source_points = np.float32([
            keypoints1[m.queryIdx].pt
            for m in good_matches
        ]).reshape(
            -1, 1, 2
        )

        destination_points = np.float32([
            keypoints2[m.trainIdx].pt
            for m in good_matches
        ]).reshape(
            -1, 1, 2
        )


        H, mask = cv2.findHomography(
            source_points,
            destination_points,
            cv2.RANSAC,
            5.0
        )


        if mask is not None:

            inliers = int(
                mask.sum()
            )


    # ----------------------------------------------------------
    # INLIER PERCENTAGE
    # ----------------------------------------------------------

    inlier_percentage = (
        inliers
        / len(good_matches)
        * 100
        if len(good_matches) > 0
        else 0
    )


    # ----------------------------------------------------------
    # PROCESSING TIME
    # ----------------------------------------------------------

    processing_time = (
        time.perf_counter()
        - start_time
    ) * 1000


    # ----------------------------------------------------------
    # RETURN RESULTS
    # ----------------------------------------------------------

    return {

        "Condition": condition,

        "Keypoints Image 1":
            len(keypoints1),

        "Keypoints Image 2":
            len(keypoints2),

        "Good Matches":
            len(good_matches),

        "Match Percentage":
            round(
                match_percentage,
                2
            ),

        "Inliers":
            inliers,

        "Inlier Percentage":
            round(
                inlier_percentage,
                2
            ),

        "Processing Time (ms)":
            round(
                processing_time,
                2
            ),

        "Keypoints1":
            keypoints1,

        "Keypoints2":
            keypoints2,

        "Good":
            good_matches
    }


# ==============================================================
# STEP 11: APPROACH 1 - WITHOUT PREPROCESSING
# ==============================================================

print("\nRunning experiment WITHOUT preprocessing...")

result_without = perform_experiment(
    gray1,
    gray2,
    "Without Preprocessing"
)


# ==============================================================
# STEP 12: APPROACH 2 - WITH PREPROCESSING
# ==============================================================

print(
    "Running experiment WITH preprocessing..."
)

result_with = perform_experiment(
    processed1,
    processed2,
    "With Preprocessing"
)


# ==============================================================
# STEP 13: CREATE RESULTS TABLE
# ==============================================================

results = [

    result_without,
    result_with

]


results_df = pd.DataFrame([

    {
        "Approach":
            r["Condition"],

        "Keypoints Image 1":
            r["Keypoints Image 1"],

        "Keypoints Image 2":
            r["Keypoints Image 2"],

        "Good Matches":
            r["Good Matches"],

        "Match Percentage (%)":
            r["Match Percentage"],

        "RANSAC Inliers":
            r["Inliers"],

        "Inlier Percentage (%)":
            r["Inlier Percentage"],

        "Processing Time (ms)":
            r["Processing Time (ms)"]
    }

    for r in results

])


# ==============================================================
# STEP 14: DISPLAY RESULTS
# ==============================================================

print("\n")
print("=" * 75)
print("        PREPROCESSING VS FEATURE EXTRACTION")
print("=" * 75)

print(
    results_df.to_string(
        index=False
    )
)


# ==============================================================
# STEP 15: DRAW SIFT KEYPOINTS
# ==============================================================

keypoints_without_1 = cv2.drawKeypoints(
    img1,
    result_without["Keypoints1"],
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

keypoints_with_1 = cv2.drawKeypoints(
    img1,
    result_with["Keypoints1"],
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)


keypoints_without_2 = cv2.drawKeypoints(
    img2,
    result_without["Keypoints2"],
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

keypoints_with_2 = cv2.drawKeypoints(
    img2,
    result_with["Keypoints2"],
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)


# ==============================================================
# STEP 16: VISUALIZE KEYPOINT COMPARISON
# ==============================================================

plt.figure(figsize=(14, 10))


plt.subplot(2, 2, 1)

plt.imshow(
    cv2.cvtColor(
        keypoints_without_1,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "Without Preprocessing - Image 1\n"
    + str(result_without["Keypoints Image 1"])
    + " Keypoints"
)

plt.axis("off")


plt.subplot(2, 2, 2)

plt.imshow(
    cv2.cvtColor(
        keypoints_with_1,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "With Preprocessing - Image 1\n"
    + str(result_with["Keypoints Image 1"])
    + " Keypoints"
)

plt.axis("off")


plt.subplot(2, 2, 3)

plt.imshow(
    cv2.cvtColor(
        keypoints_without_2,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "Without Preprocessing - Image 2\n"
    + str(result_without["Keypoints Image 2"])
    + " Keypoints"
)

plt.axis("off")


plt.subplot(2, 2, 4)

plt.imshow(
    cv2.cvtColor(
        keypoints_with_2,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "With Preprocessing - Image 2\n"
    + str(result_with["Keypoints Image 2"])
    + " Keypoints"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ==============================================================
# STEP 17: VISUALIZE GOOD MATCHES
# ==============================================================

match_image_without = cv2.drawMatches(
    img1,
    result_without["Keypoints1"],
    img2,
    result_without["Keypoints2"],
    result_without["Good"],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


match_image_with = cv2.drawMatches(
    img1,
    result_with["Keypoints1"],
    img2,
    result_with["Keypoints2"],
    result_with["Good"],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


plt.figure(figsize=(16, 10))


plt.subplot(2, 1, 1)

plt.imshow(
    cv2.cvtColor(
        match_image_without,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "Feature Matching WITHOUT Preprocessing"
)

plt.axis("off")


plt.subplot(2, 1, 2)

plt.imshow(
    cv2.cvtColor(
        match_image_with,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "Feature Matching WITH Preprocessing"
)

plt.axis("off")


plt.tight_layout()
plt.show()


# ==============================================================
# STEP 18: CALCULATE IMPROVEMENT
# ==============================================================

def calculate_improvement(
    without_value,
    with_value
):

    if without_value == 0:

        return 0

    return (
        (with_value - without_value)
        / without_value
    ) * 100


keypoint_improvement = calculate_improvement(
    (
        result_without["Keypoints Image 1"]
        + result_without["Keypoints Image 2"]
    ) / 2,

    (
        result_with["Keypoints Image 1"]
        + result_with["Keypoints Image 2"]
    ) / 2
)


match_improvement = calculate_improvement(
    result_without["Good Matches"],
    result_with["Good Matches"]
)


inlier_improvement = calculate_improvement(
    result_without["Inliers"],
    result_with["Inliers"]
)


accuracy_improvement = (
    result_with["Inlier Percentage"]
    - result_without["Inlier Percentage"]
)


time_change = calculate_improvement(
    result_without["Processing Time (ms)"],
    result_with["Processing Time (ms)"]
)


# ==============================================================
# STEP 19: DISPLAY IMPROVEMENT
# ==============================================================

print("\n")
print("=" * 60)
print("             IMPROVEMENT ANALYSIS")
print("=" * 60)

print(
    "Keypoint change:",
    round(
        keypoint_improvement,
        2
    ),
    "%"
)

print(
    "Good match change:",
    round(
        match_improvement,
        2
    ),
    "%"
)

print(
    "RANSAC inlier change:",
    round(
        inlier_improvement,
        2
    ),
    "%"
)

print(
    "Inlier percentage change:",
    round(
        accuracy_improvement,
        2
    ),
    "percentage points"
)

print(
    "Processing time change:",
    round(
        time_change,
        2
    ),
    "%"
)


# ==============================================================
# STEP 20: GRAPH - KEYPOINT COMPARISON
# ==============================================================

labels = [
    "Image 1",
    "Image 2"
]

without_keypoints = [
    result_without["Keypoints Image 1"],
    result_without["Keypoints Image 2"]
]

with_keypoints = [
    result_with["Keypoints Image 1"],
    result_with["Keypoints Image 2"]
]


x = np.arange(
    len(labels)
)

width = 0.35


plt.figure(figsize=(9, 6))

plt.bar(
    x - width / 2,
    without_keypoints,
    width,
    label="Without Preprocessing"
)

plt.bar(
    x + width / 2,
    with_keypoints,
    width,
    label="With Preprocessing"
)

plt.xticks(
    x,
    labels
)

plt.ylabel(
    "Number of Keypoints"
)

plt.title(
    "SIFT Keypoint Comparison"
)

plt.legend()

plt.tight_layout()
plt.show()


# ==============================================================
# STEP 21: GRAPH - GOOD MATCHES
# ==============================================================

plt.figure(figsize=(8, 6))

plt.bar(
    [
        "Without\nPreprocessing",
        "With\nPreprocessing"
    ],
    [
        result_without["Good Matches"],
        result_with["Good Matches"]
    ]
)

plt.ylabel(
    "Number of Good Matches"
)

plt.title(
    "Good Feature Match Comparison"
)

plt.tight_layout()
plt.show()


# ==============================================================
# STEP 22: GRAPH - MATCH / INLIER PERCENTAGE
# ==============================================================

plt.figure(figsize=(9, 6))

approaches = [
    "Without\nPreprocessing",
    "With\nPreprocessing"
]

match_percentages = [
    result_without["Match Percentage"],
    result_with["Match Percentage"]
]

inlier_percentages = [
    result_without["Inlier Percentage"],
    result_with["Inlier Percentage"]
]


x = np.arange(
    len(approaches)
)

width = 0.35


plt.bar(
    x - width / 2,
    match_percentages,
    width,
    label="Match Percentage"
)

plt.bar(
    x + width / 2,
    inlier_percentages,
    width,
    label="Inlier Percentage"
)

plt.xticks(
    x,
    approaches
)

plt.ylabel(
    "Percentage (%)"
)

plt.title(
    "Feature Matching Quality Comparison"
)

plt.legend()

plt.tight_layout()
plt.show()


# ==============================================================
# STEP 23: PROCESSING TIME COMPARISON
# ==============================================================

plt.figure(figsize=(8, 6))

plt.bar(
    [
        "Without\nPreprocessing",
        "With\nPreprocessing"
    ],
    [
        result_without["Processing Time (ms)"],
        result_with["Processing Time (ms)"]
    ]
)

plt.ylabel(
    "Processing Time (ms)"
)

plt.title(
    "Overall Processing Time Comparison"
)

plt.tight_layout()
plt.show()


# ==============================================================
# STEP 24: SAVE RESULTS
# ==============================================================

results_df.to_csv(
    "preprocessing_vs_feature_extraction.csv",
    index=False
)

cv2.imwrite(
    "matches_without_preprocessing.jpg",
    match_image_without
)

cv2.imwrite(
    "matches_with_preprocessing.jpg",
    match_image_with
)


print("\n")
print("=" * 60)
print("             FILES SAVED")
print("=" * 60)

print(
    "1. preprocessing_vs_feature_extraction.csv"
)

print(
    "2. matches_without_preprocessing.jpg"
)

print(
    "3. matches_with_preprocessing.jpg"
)


# ==============================================================
# STEP 25: FINAL CONCLUSION
# ==============================================================

print("\n")
print("=" * 70)
print("                    CONCLUSION")
print("=" * 70)

print("""
The experiment compared SIFT feature extraction with
and without image preprocessing.

""")
