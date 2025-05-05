import os
import requests
import json

# Replace with your API Key
API_KEY = "Your_API_Key"
MODEL_ID = "dota-oiell"
VERSION = "10"
CONFIDENCE_THRESHOLD = 0.05 # Adjust this value as needed
URL = f"https://detect.roboflow.com/{MODEL_ID}/{VERSION}?api_key={API_KEY}"

# Parent directory containing multiple video folders
parent_folder = r"/home/magecliff/Traffic_Recognition/Carom3/imgs"
output_folder = r"/home/magecliff/Traffic_Recognition/Carom3/jsons2"

os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

def process_image(image_path):
    """Sends image to Roboflow API and retrieves bounding boxes."""
    with open(image_path, "rb") as f:
        image_data = f.read()

    response = requests.post(URL, files={"file": image_data})

    # Handle API errors
    try:
        result = response.json()
    except json.JSONDecodeError:
        print(f"Error decoding response for {image_path}")
        return [{"bbox": [0, 0, 0, 0], "class_id": -1}]

    detections = result.get("predictions", [])

    # Convert results to required format with confidence filtering
    formatted_bboxes = []
    for det in detections:
        confidence = det.get("confidence", 0)  # Get confidence score
        if confidence < CONFIDENCE_THRESHOLD:  # Skip low-confidence detections
            continue

        x_min = int(det["x"] - det["width"] / 2)
        y_min = int(det["y"] - det["height"] / 2)
        x_max = int(det["x"] + det["width"] / 2)
        y_max = int(det["y"] + det["height"] / 2)
        class_name = det["class"]

        formatted_bboxes.append({
            "bbox": [x_min, y_min, x_max, y_max],
            "class_id": class_name
        })

    # If no detections, return empty bbox
    if not formatted_bboxes:
        return [{"bbox": [0, 0, 0, 0], "class_id": -1}]

    return formatted_bboxes

def process_video_folder(video_folder):
    """Processes all frames in a video folder and saves bbox data into a separate output JSON folder."""

    video_name = os.path.basename(video_folder)
    json_path = os.path.join(output_folder, f"{video_name}.json")

    # Check if the JSON file already exists, if so, skip processing
    if os.path.exists(json_path):
        print(f"Skipping {video_name}, JSON file already exists.")
        return

    print(f"Processing video: {video_name}...")
    video_data = []  # List to store bounding box info for all frames

    for filename in sorted(os.listdir(video_folder)):  # Sort ensures proper frame order
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(video_folder, filename)
            print(f"Processing {filename}...")

            # Get bounding boxes
            bbox_data = process_image(image_path)

            # Append bounding boxes as a list inside another list
            video_data.append([bbox_data])

    # Save bbox data as a single JSON file in the separate output folder
    with open(json_path, "w") as json_file:
        json.dump(video_data, json_file, indent=4)

# Iterate over all video folders
for video_folder in sorted(os.listdir(parent_folder)):  # Sort ensures correct order
    full_video_path = os.path.join(parent_folder, video_folder)

    if os.path.isdir(full_video_path):  # Ensure it's a directory
        process_video_folder(full_video_path)

print(f"\nProcessing complete. All bounding boxes saved in: {output_folder}")
