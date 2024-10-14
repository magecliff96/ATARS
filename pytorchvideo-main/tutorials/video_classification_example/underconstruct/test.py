import os
import torch
import pytorchvideo.models as models
import pytorchvideo.data
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from torchvision.io import read_video
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from traffic_dataset import *

def load_custom_resnet_model(model_path):
    # Load your custom ResNet model architecture
    model = models.resnet.create_resnet(
                input_channel=3,
                model_num_class=39,
            ).to(device="cuda")
    state_dict = torch.load(model_path)

    modified_state_dict = {}
    prefix_to_remove = "model."
    for key, value in state_dict['state_dict'].items():
        new_key = key.replace(prefix_to_remove, "")
        modified_state_dict[new_key] = value
        
    model.load_state_dict(modified_state_dict)
    return model

class LimitDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos

def make_transforms(mode: str):
    transform = [
        video_transform(mode),
        RemoveKey("audio"),
    ]
    return Compose(transform)

def video_transform(mode: str):
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(8),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
            ]
            + (
                [
                    RandomShortSideScale(
                        min_size=256,
                        max_size=320,
                    ),
                    RandomCrop(224),
                    RandomHorizontalFlip(p=0.5),
                ]
                if mode == "train"
                else [
                    ShortSideScale(256),
                    CenterCrop(224),
                ]
            )
        ),
    )

def main():
    # Folder path containing the videos
    data_path = r"D:/research/traffic/UAV-benchmark-M"
    video_path_prefix = r"D:/research/traffic/Traffic_Recognition/frame"

    # Path to your custom-trained ResNet model
    custom_model_path = r"D:/research/traffic/Traffic_Recognition/pytorchvideo-main/tutorials/video_classification_example/log/model.ckpt"

    # Load the custom-trained ResNet model
    custom_resnet_model = load_custom_resnet_model(custom_model_path)

    sampler = SequentialSampler
    val_transform = make_transforms(mode="val")
    val_dataset = Trafficdataloader(
        data_path=os.path.join(data_path, "val.csv"),
        clip_sampler=pytorchvideo.data.make_clip_sampler(
            "uniform", 3
        ),
        video_path_prefix=video_path_prefix,
        transform=val_transform,
        video_sampler=sampler,
    )
    data_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=8,
    )


    # Perform inference on the videos
    custom_resnet_model.eval()
    for i, batch in enumerate(data_loader):
        video, fps = batch

        # Preprocess the video frames to match the input format expected by PyTorchVideo models
        video = video.permute(0, 3, 1, 2)  # Permute dimensions (batch, time, height, width) -> (batch, channels, time, height, width)

        outputs = custom_resnet_model(video)

        # Process the outputs as needed
        # For example, you can get the predicted labels
        predicted_labels = torch.argmax(outputs, dim=1).item()
        print(f"Video {i+1}: Predicted Label - {predicted_labels}")

if __name__ == "__main__":
    main()