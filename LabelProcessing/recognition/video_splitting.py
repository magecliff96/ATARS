from moviepy.editor import VideoFileClip
import os

def split_video_into_pieces(video_path, output_dir, num_pieces=10):
    # Load the video
    video = VideoFileClip(video_path)
    
    # Get the duration of the video
    duration = video.duration
    
    # Calculate the duration of each piece
    piece_duration = duration / num_pieces
    
    for i in range(num_pieces):
        # Calculate the start and end times for each piece
        start_time = i * piece_duration
        end_time = (i + 1) * piece_duration
        
        # Extract the subclip
        subclip = video.subclip(start_time, end_time)
        
        # Save the subclip
        output_path = f"{output_dir}/piece_{i+1}.mp4"
        subclip.write_videofile(output_path, codec='libx264')
    
    print(f"Video has been split into {num_pieces} pieces.")

# Example usage
video_dir = r'D:\research\traffic\CAROM_Air\dataset'
video_filename = '1000_7_3'
video_path = os.path.join(video_dir, f'{video_filename}.mp4')
output_dir = r'D:\research\traffic\CAROM_Air\split_video'
split_video_into_pieces(video_path, output_dir)
