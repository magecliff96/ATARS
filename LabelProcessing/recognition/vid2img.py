import os
import cv2

def extract_frames_from_videos(input_videos_path, output_images_path):
    """
    Converts videos into folders of images.

    Parameters:
        input_videos_path (str): Path to the folder containing the input videos.
        output_images_path (str): Path to the root folder where images will be saved.
    """
    # Ensure output directory exists
    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    # Iterate over all video files in the input folder
    for video_file in os.listdir(input_videos_path):
        video_path = os.path.join(input_videos_path, video_file)

        # Skip non-video files
        if not os.path.isfile(video_path):
            continue

        # Extract the video name (without extension)
        video_name = os.path.splitext(video_file)[0]

        # Create a folder for the video frames if it doesn't exist
        video_output_path = os.path.join(output_images_path, video_name)
        if not os.path.exists(video_output_path):
            os.makedirs(video_output_path)
        else:
            print(f"Output folder for {video_name} already exists. Skipping...")
            continue

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_file}")
            continue

        frame_count = 0

        # Read and save frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save the frame as an image
            frame_filename = os.path.join(video_output_path, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)

            frame_count += 1

        cap.release()
        print(f"Extracted {frame_count} frames from {video_file} to {video_output_path}")


if __name__ == "__main__":
    # Input folder containing videos
    input_videos_folder = r"/home/magecliff/Traffic_Recognition/Carom3/og"

    # Output folder to save images
    output_images_folder = r"/home/magecliff/Traffic_Recognition/Carom3/imgs"

    # Run the frame extraction
    extract_frames_from_videos(input_videos_folder, output_images_folder)
