import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import subprocess

# Define the path to the input Excel file and the directory containing the main videos
intervals_file_path = r'D:\research\traffic\label_tool\sampled_intervals.xlsx'
videos_directory = r'D:\research\traffic\label_tool'
output_directory = r'D:\research\traffic\label_tool\clips'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Load the intervals from the Excel file
xls = pd.ExcelFile(intervals_file_path)
sheet_names = xls.sheet_names

# Function to extract clips using ffmpeg directly to handle the codec issue
def extract_clip(video_path, start_time, end_time, output_path):
    command = [
        'ffmpeg',
        '-i', video_path,
        '-ss', str(start_time),
        '-to', str(end_time),
        '-c', 'copy',
        '-map', '0:v',  # Map only the video stream
        output_path
    ]
    subprocess.run(command, check=True)

# Process each sheet
for sheet in sheet_names:
    if sheet not in ['outline', 'template']:
        print(f"Processing sheet: {sheet}")
        df = pd.read_excel(intervals_file_path, sheet_name=sheet)
        
        # Extract the start and end frames and convert to seconds (assuming fps is 30)
        fps = 30
        intervals = df[['clip_name', 'start_frame', 'end_frame']].dropna()
        for index, row in intervals.iterrows():
            start_time = row['start_frame'] / fps
            end_time = row['end_frame'] / fps
            output_clip_path = os.path.join(output_directory, f"{row['clip_name']}.mp4")
            
            # Path to the main video
            main_video_path = os.path.join(videos_directory, f"{sheet}.mp4")
            
            # Extract the clip
            extract_clip(main_video_path, start_time, end_time, output_clip_path)
            print(f"Extracted clip {row['clip_name']} from {sheet}: {output_clip_path}")