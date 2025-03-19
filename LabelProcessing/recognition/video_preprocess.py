import cv2
import os

def process_video(video_path, output_video_path, output_dir='processed_frames'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("無法打開影片檔案")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"影片FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(output_filename, frame)

        cv2.putText(frame, f"Frame {frame_count}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5, cv2.LINE_AA)

        out.write(frame)
        frame_count += 1

        if frame_count%25 == 0: #to make sure the code is running
            print(frame_count)

    cap.release()
    out.release()

video_dir = r'D:\research\traffic\CAROM_Air\dataset'
video_filename = '1000_7_3'
video_path = os.path.join(video_dir, f'{video_filename}.mp4')
output_dir = r'D:\research\traffic\CAROM_Air\dataset_frame'
output_path = os.path.join(output_dir, video_filename)
output_video_dir = r'D:\research\traffic\CAROM_Air\dataset_processed'
output_video = os.path.join(output_video_dir, f'{video_filename}.mp4')

process_video(video_path, output_video, output_path)