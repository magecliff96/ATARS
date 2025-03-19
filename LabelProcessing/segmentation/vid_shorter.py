import os
import cv2
import pandas as pd

def short_side_resize(frame, target_size=256):
    h, w, _ = frame.shape
    
    # Determine the scale factor
    if h < w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))

    # Resize the frame
    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame

def shifted_center_crop(frame, crop_size=256, shift=0):
    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2
    
    # Apply a slight shift to the center
    if w > crop_size + 2 * shift:
        center_x += shift  # Shift along the width

    # Calculate the cropping box
    x1 = max(0, center_x - crop_size // 2)
    y1 = max(0, center_y - crop_size // 2)
    
    # Crop to 256x256
    cropped_frame = frame[y1:y1+crop_size, x1:x1+crop_size]
    return cropped_frame


def process_videos_in_folder(folder_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.MP4', '.mp4', '.avi', '.mov', '.mkv')):  # Add more formats if needed
            video_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create a VideoWriter object to save the resized video
            output_path = os.path.join(output_folder, f"{filename}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (256, 256))
            
            frame_count = 0
            #while frame_count < min(512, total_frames):
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize the short side to 256
                resized_frame = short_side_resize(frame, target_size=256)

                # Check if video name starts with '1000' for shifted center cropping
                if filename.startswith('1000'):
                    resized_frame = shifted_center_crop(resized_frame, crop_size=256, shift=64)
                else:
                    resized_frame = shifted_center_crop(resized_frame, crop_size=256)

                # Write the frame to the output video
                out.write(resized_frame)

                frame_count += 1
            
            # Release everything
            cap.release()
            out.release()
            print(f"Saved processed video to {output_path}")
        else:
            print(f"Skipping: {filename} (not a video file)")

# Usage example
folder_path = r'D:\research\traffic\Traffic_Recognition\CAROM_Air\label_tool\videos'
output_folder = r'D:\research\traffic\Traffic_Recognition\Carom_TempSeg\videos'
process_videos_in_folder(folder_path, output_folder)