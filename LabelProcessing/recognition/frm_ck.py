# import cv2
# import os

# def trim_video_to_32_frames(video_path, output_folder):
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)

#     # Load the video
#     cap = cv2.VideoCapture(video_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#     # Read all frames
#     frames = []
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)

#     cap.release()

#     # If more than 32 frames, trim the extra frames from the end
#     if frame_count > 32:
#         frames = frames[:32]  # Keep only the first 32 frames

#     # Save the processed video
#     output_path = os.path.join(output_folder, os.path.basename(video_path))
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     for frame in frames:
#         out.write(frame)

#     out.release()
#     print(f"Processed video saved at: {output_path}")

# def process_videos_in_folder(input_folder, output_folder):
#     # Get all video files in the folder
#     video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

#     if not video_files:
#         print("No videos found in the input folder.")
#         return

#     for video_file in video_files:
#         video_path = os.path.join(input_folder, video_file)
#         print(f"Processing: {video_path}")
#         trim_video_to_32_frames(video_path, output_folder)

# # Change these paths accordingly
# input_folder = r"/home/magecliff/Traffic_Recognition/Carom3/masks"
# output_folder = r"/home/magecliff/Traffic_Recognition/Carom3/masks_n"
# process_videos_in_folder(input_folder, output_folder)



# import cv2
# import os

# def check_video_frames(input_folder):
#     # Get all video files in the folder
#     video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

#     if not video_files:
#         print("No videos found in the input folder.")
#         return

#     for video_file in video_files:
#         video_path = os.path.join(input_folder, video_file)

#         # Open the video file
#         cap = cv2.VideoCapture(video_path)
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
#         # Read the first frame to get the number of channels
#         ret, frame = cap.read()
#         cap.release()

#         if ret:
#             channels = frame.shape[2] if len(frame.shape) == 3 else 1  # Check if color or grayscale
#             print(f"Video: {video_file}, Frames: {frame_count}, Shape: ({frame_count}, {height}, {width}, {channels})")
#         else:
#             print(f"Video: {video_file}, Frames: {frame_count}, Shape: Could not read first frame")

# # Example usage
# input_folder = r"/home/magecliff/Traffic_Recognition/Carom3/masks"  # Change this to your video folder
# check_video_frames(input_folder)

import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F

def downscale_with_torch(video_path, output_folder, new_width, new_height, threshold_value=0.5):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure the video has exactly 32 frames
    if total_frames != 32:
        print(f"❌ Error: {video_path} does not have 32 frames (found {total_frames}) - Skipping.")
        cap.release()
        return

    # Define output video path
    output_path = os.path.join(output_folder, os.path.basename(video_path))
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale and invert colors
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inverted_frame = cv2.bitwise_not(gray_frame)

        # Convert frame to tensor and normalize to [0,1]
        tensor_frame = torch.tensor(inverted_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0



        # Resize to 14x14 using nearest interpolation
        resized = F.interpolate(tensor_frame, size=(new_width, new_width), mode='nearest')

        # Apply thresholding (convert to 0 or 1)
        binary_tensor = (resized > threshold_value).float()
        # Convert back to numpy (uint8)
        binary_frame = (binary_tensor.numpy().squeeze(0).squeeze(0) * 255).astype(np.uint8)
        # Convert to 3-channel grayscale for video writing
        binary_frame = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)
        out.write(binary_frame)
        processed_frames += 1

    cap.release()
    out.release()

    # Ensure output has exactly 32 frames
    cap_out = cv2.VideoCapture(output_path)
    output_frame_count = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
    output_width = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_out.release()

    if output_frame_count != 32 or output_width != new_width or output_height != new_height:
        print(f"❌ Error: Processed video {output_path} has incorrect dimensions ({output_width}x{output_height}) or frame count ({output_frame_count}). Deleting...")
        os.remove(output_path)
    else:
        print(f"✅ Successfully processed: {output_path} ({output_width}x{output_height}, {output_frame_count} frames)")

def process_videos_in_folder(input_folder, output_folder, new_width, new_height, threshold_value=0.5):
    # Get all video files in the folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files:
        print("No videos found in the input folder.")
        return

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        print(f"Processing: {video_path}")
        downscale_with_torch(video_path, output_folder, new_width, new_height, threshold_value)

# Change these paths accordingly
input_folder = r"/home/magecliff/Traffic_Recognition/Carom3/masks_og"
output_folder = r"/home/magecliff/Traffic_Recognition/Carom3/masks"

process_videos_in_folder(input_folder, output_folder, new_width=28, new_height=28) #must be even
