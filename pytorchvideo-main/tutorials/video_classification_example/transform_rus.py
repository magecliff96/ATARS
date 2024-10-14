import torch
import cv2
import os
import numpy as np
from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.encoded_video import EncodedVideo
import torchvision.transforms as T
from torchvision.io import read_video, write_video
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)


class ShiftedCenterCrop:
    def __init__(self, size, shift_x=0, shift_y=0):
        self.size = size
        self.shift_x = shift_x
        self.shift_y = shift_y

    def __call__(self, video):
        #print(video.shape)
        _, _, h, w = video.shape
        crop_h, crop_w = self.size

        # Ensure crop size does not exceed the original dimensions
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)

        # Calculate new center with shift
        center_x = w // 2 + self.shift_x
        center_y = h // 2 + self.shift_y

        # Calculate cropping coordinates
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # Ensure coordinates are within bounds
        x1 = min(w - crop_w, x1)
        y1 = min(h - crop_h, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Adjust for even dimensions
        if (x2 - x1) % 2 != 0:
            x2 -= 1
        if (y2 - y1) % 2 != 0:
            y2 -= 1

        return video[:, :, y1:y2, x1:x2]


class VideoPreprocessor:
    def __init__(self, path_order_cache=None):
        self.path_order_cache = path_order_cache

    def video_from_path(
            self, filepath, decode_video=True, decode_audio=False, decoder="pyav", fps=30
        ):
            try:
                is_file = g_pathmgr.isfile(filepath)
                is_dir = g_pathmgr.isdir(filepath)
            except NotImplementedError:

                # Not all PathManager handlers support is{file,dir} functions, when this is the
                # case, we default to assuming the path is a file.
                is_file = True
                is_dir = False

            if is_file:
                return EncodedVideo.from_path(
                    filepath,
                    decode_video=decode_video,
                    decode_audio=decode_audio,
                    decoder=decoder,
                )
            elif is_dir:
                from pytorchvideo.data.frame_video import FrameVideo

                assert not decode_audio, "decode_audio must be False when using FrameVideo"
                return FrameVideo.from_directory(
                    filepath, fps, path_order_cache=self.path_order_cache
                )
            else:
                raise FileNotFoundError(f"{filepath} not found.")

    def preprocess_video_pytorch(self, video_path, output_path, args, use_shifted_crop=False):
        video_data = self.video_from_path(video_path, decode_video=True, decode_audio=False, decoder="pyav", fps=30)

        # Access video frames as a tensor
        video_tensor = video_data.get_clip(start_sec=0, end_sec=video_data.duration)['video']
        
        # Check if the video tensor is empty
        if video_tensor.nelement() == 0:
            raise ValueError(f"Video {video_path} did not load correctly, resulting in an empty tensor.")
        
        print(f"Processing: {video_path}")
        print(f'Video shape before transformation: {video_tensor.shape}')
        
        # Define transformations
        if use_shifted_crop:
            transforms = T.Compose([
                UniformTemporalSubsample(args.video_num_subsampled),
                ShortSideScale(size=args.video_min_short_side_scale),
                ShiftedCenterCrop(size=(args.video_crop_size, args.video_crop_size), shift_x=args.shift_x, shift_y=args.shift_y),
            ])
        else:
            transforms = T.Compose([
                UniformTemporalSubsample(args.video_num_subsampled),
                ShortSideScale(size=args.video_min_short_side_scale),
                CenterCrop(size=args.video_crop_size),
            ])
        
        # Apply transformations
        transformed_video = transforms(video_tensor)
        
        # Convert back to uint8
        transformed_video = (transformed_video).byte().permute(1, 2, 3, 0)
        print(f'Video shape after transformation: {transformed_video.shape}')
        
        # Save video
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_video(output_path, transformed_video, fps=30)

    def preprocess_videos_in_directory(self, directory_path, output_directory, args):
        for filename in os.listdir(directory_path):
            if filename.endswith(('.mp4', '.avi')):  # You can add more file extensions as needed
                video_path = os.path.join(directory_path, filename)
                output_path = os.path.join(output_directory, filename)
                use_shifted_crop = filename.startswith("1000")
                try:
                    self.preprocess_video_pytorch(video_path, output_path, args, use_shifted_crop)
                    print(f"Processed and saved: {output_path}")
                except ValueError as e:
                    print(f"Skipping {video_path}: {e}")

# Example usage
class Args:
    def __init__(self):
        self.video_num_subsampled = 32
        self.video_means = [0.45, 0.45, 0.45]
        self.video_stds = [0.225, 0.225, 0.225]
        self.video_min_short_side_scale = 256
        self.video_max_short_side_scale = 320
        self.video_crop_size = 224
        self.shift_x = 64  # Example shift values
        self.shift_y = -16

def main():
    args = Args()
    input_video_path = r"/home/magecliff/Traffic_Recognition/Carom3/original"
    output_video_path = r"/home/magecliff/Traffic_Recognition/Carom3/videos"
    preprocessor = VideoPreprocessor()
    preprocessor.preprocess_videos_in_directory(input_video_path, output_video_path, args)

if __name__ == "__main__":
    main()
