# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import annotations
import time
import numpy as np
import gc
import logging
import ast
import os
import pathlib
import itertools
import torch
import torch.utils.data

from pytorchvideo.data.clip_sampling import ClipSampler
#from pytorchvideo.data.video import VideoPathHandler #videopathhandler
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from iopath.common.file_io import g_pathmgr
from torchvision.datasets.folder import make_dataset


logger = logging.getLogger(__name__)


def Trafficdataloader(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
) -> TrafficDataset:

    video_paths = TrafficVideoPaths.from_path(data_path)
    video_paths.path_prefix = video_path_prefix
    dataset = TrafficDataset(
        video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )
    return dataset

class TrafficDataset(torch.utils.data.IterableDataset):

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        video_paths: List[Tuple[str, Optional[dict]]],
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = True,
        decode_video: bool = True,
        decoder: str = "pyav",
    ) -> None:
        self._decode_audio = decode_audio
        self._decode_video = decode_video
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = video_paths
        self._decoder = decoder

        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None
        self._loaded_video_label = None
        self._loaded_clip = None
        self._last_clip_end_time = None
        self.video_path_handler = VideoPathHandler()


    @property
    def video_sampler(self):
        return self._video_sampler

    @property
    def num_videos(self):
        return len(self.video_sampler)

    def __len__(self): 
        return len(self.video_sampler)

    def __next__(self) -> dict: ### main issue
        if not self._video_sampler_iter:
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES): #0 iter per loop but is ran by trainer many times
            # Reuse previously stored video if there are still clips to be sampled from
            # the last loaded video.
            if self._loaded_video_label:             #not neckbelow
                video, info_dict, video_index = self._loaded_video_label
            else:
                video_index = next(self._video_sampler_iter)
                try:
                    video_path, info_dict = self._labeled_videos[video_index]
                    video = self.video_path_handler.video_from_path(
                        video_path,
                        decode_audio=self._decode_audio,
                        decode_video=self._decode_video, 
                        decoder=self._decoder,
                    )
                    self._loaded_video_label = (video, info_dict, video_index)
                except Exception as e:
                    logger.debug(
                        "Failed to load video with error: {}; trial {}".format(
                            e,
                            i_try,
                        )
                    )
                    logger.exception("Video load exception")
                    continue
            # not neck above

            (
                clip_start,
                clip_end,
                clip_index,
                aug_index,
                is_last_clip,
            ) = self._clip_sampler(self._last_clip_end_time, video.duration, info_dict)


            start_time = time.time() #15 secs
            #neck
            if isinstance(clip_start, list):  # multi-clip in each sample
                if aug_index[0] == 0:
                    self._loaded_clip = {}
                    loaded_clip_list = []
                    for i in range(len(clip_start)):
                        clip_dict = video.get_clip(clip_start[i], clip_end[i])
                        if clip_dict is None or clip_dict["video"] is None:
                            self._loaded_clip = None
                            break
                        loaded_clip_list.append(clip_dict)

                    if self._loaded_clip is not None:
                        for key in loaded_clip_list[0].keys():
                            self._loaded_clip[key] = [x[key] for x in loaded_clip_list]

            else:  
                if aug_index == 0:
                    self._loaded_clip = video.get_clip(clip_start, clip_end)
            #neck
            end_time = time.time()
            elapsed_time = end_time - start_time
            #print(f"isinstance time taken: {elapsed_time} seconds")


            #2 sec below
            self._last_clip_end_time = clip_end

            video_is_null = (
                self._loaded_clip is None or self._loaded_clip["video"] is None
            )
            if (
                is_last_clip[-1] if isinstance(is_last_clip, list) else is_last_clip
            ) or video_is_null:

                self._loaded_video_label[0].close()
                self._loaded_video_label = None
                self._last_clip_end_time = None
                self._clip_sampler.reset()
                gc.collect()

                if video_is_null:
                    logger.debug(
                        "Failed to load clip {}; trial {}".format(video.name, i_try)
                    )
                    continue
            #2 sec above

            #not neck below
            frames = self._loaded_clip["video"]
            audio_samples = self._loaded_clip["audio"]
            sample_dict = {
                "video": frames,
                "video_name": video.name,
                "video_index": video_index,
                "clip_index": clip_index,
                "aug_index": aug_index,
                **info_dict,
                **({"audio": audio_samples} if audio_samples is not None else {}),
            }
            #notneck above

            start_time = time.time() #15 secs
            #neck
            if self._transform is not None: 
                sample_dict = self._transform(sample_dict)
                # User can force dataset to continue by returning None in transform.
                if sample_dict is None:
                    end_time = time.time()
                    elapsed_time = end_time - start_time5
                    #print(f"transfrm time taken: {elapsed_time} seconds")
                    continue
            #neck
            end_time = time.time()
            elapsed_time = end_time - start_time
            #print(f"transfrm time taken: {elapsed_time} seconds")


            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self


class TrafficVideoPaths:

    @classmethod
    def from_path(cls, data_path: str) -> TrafficVideoPaths:
        if g_pathmgr.isfile(data_path):
            return TrafficVideoPaths.from_csv(data_path)
        elif g_pathmgr.isdir(data_path):
            return TrafficVideoPaths.from_directory(data_path)
        else:
            raise FileNotFoundError(f"{data_path} not found.")

    @classmethod
    def from_csv(cls, file_path: str) -> TrafficVideoPaths:

        assert g_pathmgr.exists(file_path), f"{file_path} not found."
        video_paths_and_label = []
        with g_pathmgr.open(file_path, "r") as f:
            for path_label in f.read().splitlines():
                line_split = path_label.rsplit(None, 1)

                # The video path file may not contain labels (e.g. for a test split). We
                # assume this is the case if only 1 path is found and set the label to
                # -1 if so.
                if len(line_split) == 1:
                    file_path = line_split[0]
                    label = -1
                else:
                    file_path, label = line_split

                video_paths_and_label.append((file_path, ast.literal_eval(label)))

        assert (
            len(video_paths_and_label) > 0
        ), f"Failed to load dataset from {file_path}."
        return cls(video_paths_and_label)

    @classmethod
    def from_directory(cls, dir_path: str) -> TrafficVideoPaths:

        assert g_pathmgr.exists(dir_path), f"{dir_path} not found."

        # Find all classes based on directory names. These classes are then sorted and indexed
        # from 0 to the number of classes.
        classes = sorted(
            (f.name for f in pathlib.Path(dir_path).iterdir() if f.is_dir())
        )
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        video_paths_and_label = make_dataset(
            dir_path, class_to_idx, extensions=("mp4", "avi")
        )
        assert (
            len(video_paths_and_label) > 0
        ), f"Failed to load dataset from {dir_path}."
        return cls(video_paths_and_label)

    def __init__(
        self, paths_and_labels: List[Tuple[str, Optional[int]]], path_prefix=""
    ) -> None:
        self._paths_and_labels = paths_and_labels
        self._path_prefix = path_prefix

    def path_prefix(self, prefix):
        self._path_prefix = prefix

    path_prefix = property(None, path_prefix)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        path, label = self._paths_and_labels[index]
        return (os.path.join(self._path_prefix, path), {"label": label})

    def __len__(self) -> int:
        return len(self._paths_and_labels)


class MultiProcessSampler(torch.utils.data.Sampler): #not neck
    def __init__(self, sampler: torch.utils.data.Sampler) -> None:
        self._sampler = sampler

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers != 0:

            # Split sampler indexes by worker.
            video_indexes = range(len(self._sampler))
            worker_splits = np.array_split(video_indexes, worker_info.num_workers)
            worker_id = worker_info.id
            worker_split = worker_splits[worker_id]
            if len(worker_split) == 0:
                logger.warning(
                    f"More data workers({worker_info.num_workers}) than videos"
                    f"({len(self._sampler)}). For optimal use of processes "
                    "reduce num_workers."
                )
                return iter(())

            iter_start = worker_split[0]
            iter_end = worker_split[-1] + 1
            worker_sampler = itertools.islice(iter(self._sampler), iter_start, iter_end)
        else:

            # If no worker processes found, we return the full sampler.
            worker_sampler = iter(self._sampler)
        return worker_sampler


#video_path_handler
class VideoPathHandler:
    """
    Utility class that handles all deciphering and caching of video paths for
    encoded and frame videos.
    """

    def __init__(self) -> None:
        # Pathmanager isn't guaranteed to be in correct order,
        # sorting is expensive, so we cache paths in case of frame video and reuse.
        self.path_order_cache = {}

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
            from pytorchvideo.data.encoded_video import EncodedVideo

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

