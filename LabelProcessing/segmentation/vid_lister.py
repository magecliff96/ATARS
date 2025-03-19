import os

def generate_video_list(folder_path, output_file="video_list.txt"):
    # Open the output file in write mode
    with open(output_file, 'w') as f:
        # Walk through the folder and subfolders
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Check if the file is a video based on its extension
                if file.endswith(('.MP4','.mp4', '.avi', '.mov', '.mkv', '.flv')):
                    # Get the full path to the video file
                    full_path = os.path.join(root, file)
                    # Write the full path to the output file
                    f.write(full_path + '\n')

    print(f"Video list written to {output_file}")

# Example usage:
# Change '/path/to/videos/' to the path of your folder containing video files
folder_path = r'/home/magecliff/Traffic_Recognition/Carom_TempSeg/videos'  # Path to the folder containing video files

generate_video_list(folder_path)


