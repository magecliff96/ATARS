import cv2
import os

# Specify the folder path containing the mp4 files
folder_path = './optical/'  # Adjust this path to your video folder

# Get all mp4 files in the folder
mp4_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

# Initialize a list to track videos that had to be shortened
modified_videos = []

# Function to write frames back to a new video file
def save_truncated_video(original_video_path, output_video_path, frames):
    # Get the video properties
    cap = cv2.VideoCapture(original_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a VideoWriter to save the new video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Write the first 32 frames to the output file
    for frame in frames[:32]:
        out.write(frame)

    out.release()

# Loop through the mp4 files
for file in mp4_files:
    file_path = os.path.join(folder_path, file)
    
    # Open the video file
    cap = cv2.VideoCapture(file_path)

    # Count the number of frames in the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Collect frames if needed
    frames = []
    
    if frame_count > 32:
        # If the video has more than 32 frames, store the first 32 frames
        for i in range(32):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        # Save the shortened video (overwrite the original video)
        output_video_path = os.path.join(folder_path, f"{file}")
        save_truncated_video(file_path, output_video_path, frames)

        # Track that this video was shortened
        modified_videos.append(file)
    
    # Release the video capture object
    cap.release()

# Print the names of videos that were modified
if modified_videos:
    print("The following videos were shortened to 32 frames:")
    for video in modified_videos:
        print(video)
else:
    print("All videos already have 32 frames or fewer.")
