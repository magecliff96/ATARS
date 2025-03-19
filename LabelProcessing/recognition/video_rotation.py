import os
from moviepy.editor import VideoFileClip, vfx

def rotate_video(video_path, degrees, output_folder):
    clip = VideoFileClip(video_path)
    rotated_clip = clip.rotate(degrees)
    abs_deg = -degrees
    # Build the new filename with the desired output folder
    base_name = os.path.basename(video_path)
    new_filename = f"{base_name[:-4]}_{abs_deg}.mp4"
    output_path = os.path.join(output_folder, new_filename)
    rotated_clip.write_videofile(output_path, codec='libx264', audio_codec=False, bitrate="5000k")

def flip_video_horizontally(video_path, output_folder):
    clip = VideoFileClip(video_path)
    flipped_clip = clip.fx(vfx.mirror_x)
    # Build the new filename with the desired output folder
    base_name = os.path.basename(video_path)
    new_filename = f"{base_name[:-4]}_f.mp4"
    output_path = os.path.join(output_folder, new_filename)
    flipped_clip.write_videofile(output_path, codec='libx264', audio_codec=False, bitrate="5000k")

def rotate_videos_in_folder(folder_path, output_folder):
    for filename in os.listdir(folder_path):
        if filename.endswith((".mp4", ".MP4")):
            video_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            
            # Call the desired operations with the output folder as an argument
            rotate_video(video_path, -90, output_folder)
            rotate_video(video_path, -180, output_folder)
            rotate_video(video_path, -270, output_folder)  # Example operation
            # Add other operations as needed

            print(f"Finished processing {filename}.")

def flip_videos_in_folder(folder_path, output_folder):
    for filename in os.listdir(folder_path):
        if filename.endswith((".mp4", ".MP4")):
            video_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            
            # Call the desired operations with the output folder as an argument
            flip_video_horizontally(video_path, output_folder)
            # Add other operations as needed

            print(f"Finished processing {filename}.")

#Specify the path to the folder containing your videos and the output folder
# input_folder_rotate = r'/home/magecliff/Traffic_Recognition/Carom3/mask'
# output_folder_rotate = r'/home/magecliff/Traffic_Recognition/Carom3/mask_rt'
# rotate_videos_in_folder(input_folder_rotate, output_folder_rotate)

input_folder_flip = r'/home/magecliff/Traffic_Recognition/Carom3/mask'
output_folder_flip = r'/home/magecliff/Traffic_Recognition/Carom3/mask_f'
flip_videos_in_folder(input_folder_flip, output_folder_flip)


