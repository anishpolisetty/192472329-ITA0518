import cv2
from IPython.display import Video

# Path to the reversed video file
output_video_path = 'reverse_video.mp4'

# Display the video in the Colab output
print(f"Displaying the reversed video: {output_video_path}")
Video(output_video_path, embed=True)
# Correct video path to one available in Colab environment
video_path = "Firefly Prompt-_Create a 5-second ultra-realistic cinematic video of a single adult man walking at a.mp4"
output_video_path = "reverse_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
else:
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    # 'mp4v' for .mp4 files, 'XVID' for .avi files (might need installation for some codecs)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print("Error: Could not create video writer. Check the codec and file path.")
    else:
        frames = []
        # Read all frames into a list
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        # Write frames in reverse order
        for frame in reversed(frames):
            out.write(frame)

        print(f"Reverse video saved as '{output_video_path}'")

    cap.release()
    if out.isOpened():
        out.release()

# cv2.imshow, cv2.waitKey, and cv2.destroyAllWindows do not work in Google Colab for video playback
