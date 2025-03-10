'''
Helper Script to extract frames from thumos14 videos.

'''
import os

video_path = 'data/carom_videos'
frame_path = 'data/carom_frames'
os.system(f'mkdir {frame_path}')

subsets = ['training', 'testing']
for subset in subsets:
    num = 1
    video_list = [v for v in os.listdir(os.path.join(video_path, subset)) if v.lower().endswith('.mp4')]

    os.system(f'mkdir {frame_path}/{subset}/')
    total_num = len(video_list)
    for vid in video_list:
        print(f"[PROCESS] ({num}/{total_num}) {subset}: {vid}")
        vid = vid.split('.')[0]
        num += 1
        os.system(f'mkdir {frame_path}/{subset}/{vid}')
        os.system(f'ffmpeg -i {video_path}/{subset}/{vid}.MP4 -f image2 -vf fps=10 -s 256x256 -loglevel quiet {frame_path}/{subset}/{vid}/img_%05d.jpg')
       
