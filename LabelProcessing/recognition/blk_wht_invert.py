import cv2
import os

# Input and output folder paths
input_folder = r"/home/magecliff/Traffic_Recognition/Carom3/mask"   # Change this to your folder
output_folder = r"/home/magecliff/Traffic_Recognition/Carom3/mask_inv" # Change this to your folder

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of video files in the input folder
video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

for video_file in video_files:
    input_path = os.path.join(input_folder, video_file)
    output_path = os.path.join(output_folder, video_file)

    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Failed to open {input_path}")
        continue

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    # Define video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Invert colors (255 - pixel values)
        inverted_frame = cv2.bitwise_not(frame)

        # Write the inverted frame
        out.write(inverted_frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Processed: {video_file}")

print("All videos have been processed and saved in:", output_folder)
