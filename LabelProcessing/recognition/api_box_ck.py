
import os
import cv2
import json
import random
import numpy as np

# Paths
parent_folder = r"/home/magecliff/Traffic_Recognition/Carom3/imgs"  # Folder containing video frame folders
output_json_folder = r"/home/magecliff/Traffic_Recognition/Carom3/jsons_api"  # Folder containing JSON bbox files
output_video_folder = r"/home/magecliff/Traffic_Recognition/Carom3/example"  # Where to save the processed video

# Ensure output directory exists
os.makedirs(output_video_folder, exist_ok=True)

# Select a random JSON file
json_files = [f for f in os.listdir(output_json_folder) if f.endswith('.json')]
if not json_files:
    print("No JSON files found!")
    exit()

selected_json = random.choice(json_files)
json_file_path = os.path.join(output_json_folder, selected_json)
video_name = os.path.splitext(selected_json)[0]  # Extract video folder name

# Find corresponding video folder
video_folder_path = os.path.join(parent_folder, video_name)
if not os.path.exists(video_folder_path):
    print(f"No corresponding video folder found for {video_name}!")
    exit()

# Load bounding box data
with open(json_file_path, "r") as f:
    bbox_data = json.load(f)

# Get list of frames in the selected video folder
frame_files = sorted([f for f in os.listdir(video_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
if not frame_files:
    print(f"No image frames found in {video_name}!")
    exit()

# Read the first frame to get dimensions
first_frame = cv2.imread(os.path.join(video_folder_path, frame_files[0]))
if first_frame is None:
    print("Error reading the first frame!")
    exit()

height, width, _ = first_frame.shape

# Define video writer
output_video_path = os.path.join(output_video_folder, f"{video_name}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
fps = 10  # Adjust FPS as needed
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Color mappings for bounding boxes
colors = {
    "car": (0, 255, 0),
    "truck": (255, 0, 0),
    "bus": (0, 0, 255),
    "motorcycle": (255, 255, 0),
    "bicycle": (255, 165, 0),
    "default": (255, 255, 255)
}

# Process frames and draw bounding boxes
for idx, frame_file in enumerate(frame_files):
    frame_path = os.path.join(video_folder_path, frame_file)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Skipping corrupted frame: {frame_file}")
        continue

    # Fix: Extract inner list properly
    bboxes_list = bbox_data[idx] if idx < len(bbox_data) else []
    bboxes = bboxes_list[0] if bboxes_list else []  # Extract inner list

    for bbox_info in bboxes:
        if not isinstance(bbox_info, dict):  # Ensure bbox_info is a dictionary
            print(f"Skipping invalid bbox format in frame {frame_file}")
            continue

        bbox = bbox_info["bbox"]
        class_id = bbox_info["class_id"]

        if class_id == -1:  # Skip empty detections
            continue

        x_min, y_min, x_max, y_max = bbox
        color = colors.get(class_id, colors["default"])

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, class_id, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write frame to video
    video_writer.write(frame)


# Release video writer
video_writer.release()
print(f"\nVideo saved at: {output_video_path}")