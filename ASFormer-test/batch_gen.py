import torch
import numpy as np
import random
from grid_sampler import GridSampler, TimeWarpLayer
import torch.nn.functional as F

class BatchGenerator(object):
    def __init__(self, args, num_classes, actions_dict, gt_path, features_path, sample_rate, mode):
        self.args = args
        self.mode = mode
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.timewarp_layer = TimeWarpLayer()

        # UVAST (Optional, not used in the current context)
        self.len_seg_max = 100

        # Preloaded data will be stored here
        self.preloaded_features = []
        self.preloaded_gts = []

    def reset(self):
        self.index = 0
        self.my_shuffle()

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        self.list_of_labels = [filename.replace('.MP4', '.txt') for filename in self.list_of_examples]
        self.gts = [self.gt_path + vid for vid in self.list_of_labels]
        self.features = [self.features_path + vid.split('.')[0] + '.npy' for vid in self.list_of_examples]

        # Preload data for faster access during training
        self.preload_data()

        self.my_shuffle()

    def preload_data(self):
        """Preload features and ground truth data to avoid I/O during training."""
        for idx, vid in enumerate(self.list_of_examples):
            features = np.load(self.features[idx])  # Load feature file
            file_ptr = open(self.gts[idx], 'r')  # Read ground truth file
            content_list = file_ptr.read().split('\n')[:-1]
            file_ptr.close()

            classes = np.zeros((min(features.shape[2], len(content_list)), len(self.actions_dict)))  # Adjust for video length
            content = [item.split() for item in content_list]
            for i in range(len(classes)):
                for j in range(len(content[i])):
                    index = self.actions_dict[content[i][j]]
                    classes[i][index] = 1
            
            self.preloaded_features.append(features)
            self.preloaded_gts.append(classes)

    def my_shuffle(self):
        """Shuffle the list of examples, ground truths, and features in sync."""
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(self.list_of_examples)
        random.seed(randnum)
        random.shuffle(self.gts)
        random.seed(randnum)
        random.shuffle(self.features)

    def warp_video(self, batch_input_tensor, batch_target_tensor):
        """
        :param batch_input_tensor: (bs, C_in, L_in)
        :param batch_target_tensor: (bs, L_in)
        :return: warped input and target
        """
        bs, _, T = batch_input_tensor.shape
        grid_sampler = GridSampler(T)
        grid = grid_sampler.sample(bs)
        grid = torch.from_numpy(grid).float()

        warped_batch_input_tensor = self.timewarp_layer(batch_input_tensor, grid, mode='bilinear')
        batch_target_tensor = batch_target_tensor.unsqueeze(1).float()
        warped_batch_target_tensor = self.timewarp_layer(batch_target_tensor, grid, mode='nearest')  # no bilinear for label!
        warped_batch_target_tensor = warped_batch_target_tensor.squeeze(1).long()  # obtain the same shape

        return warped_batch_input_tensor, warped_batch_target_tensor

    def next_batch(self, batch_size, if_warp=False):
        """
        Get the next batch of data.
        :param batch_size: size of the batch
        :param if_warp: flag for data augmentation via time warping
        :return: batch data (input, target, mask)
        """
        batch = self.list_of_examples[self.index:self.index + batch_size]
        batch_gts = self.gts[self.index:self.index + batch_size]

        self.index += batch_size

        batch_input = []
        batch_target = []
        for idx, vid in enumerate(batch):
            features = self.preloaded_features[idx]
            classes = self.preloaded_gts[idx]

            batch_input.append(features)
            batch_target.append(classes)

        total_frames = []
        for i in range(len(batch_input)):
            total_frames.append(min(batch_input[i].shape[2], batch_target[i].shape[0]))
            batch_input[i] = batch_input[i][:, :, :total_frames[i]]  # Slice batch_input to the minimum temporal dimension
            batch_target[i] = batch_target[i][:total_frames[i]]

        number_of_frames = list(map(len, batch_target))
        max_length_of_frame = max([len(inner_list) for outer_list in batch_target for inner_list in outer_list])

        batch_input_tensor = torch.zeros(len(batch_input), batch_input[0].shape[1], max(number_of_frames), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(number_of_frames), max_length_of_frame, dtype=torch.float32) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(number_of_frames), dtype=torch.float)

        for i in range(len(batch_input)):
            batch_input[i] = batch_input[i].squeeze(-1).squeeze(-1)
            batch_input_tensor[i, :, :np.shape(batch_input[i])[2]] = torch.from_numpy(batch_input[i][:, :, :np.shape(batch_input[i])[2]])
            batch_target_tensor[i, :np.shape(batch_target[i])[0], :np.shape(batch_target[i])[1]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        file_name = self.list_of_labels
        data = {
            "filename": file_name,
            "input": batch_input_tensor,
            "target": batch_target_tensor,
            "mask": mask,
            "batch": batch
        }

        # If warp is applied, augment data
        if if_warp:
            data["input"], data["target"] = self.warp_video(data["input"], data["target"])

        return data
        
    #UVAST
    def convert_labels_to_segments(self, labels): # , split_segments=False, split_segments_max_dur=None
        segments = self.convert_labels(labels); segments_dict=[]
        for c in range(len(segments)):
            segment_c = segments[c]
            # we need to insert <sos> and <eos>
            segment_c.insert(0, (torch.tensor(-2, device=labels.device), -1, -1))
            segment_c.append((torch.tensor(-1, device=labels.device), segment_c[-1][-1], segment_c[-1][-1]))
            if self.args.split_segments and   self.mode == 'train' and self.args.split_segments_max_dur: 
                max_dur = self.args.split_segments_max_dur # it used to be random.sample(split_segments_max_dur, 1)[0]
                segment_c = self.split_segments_into_chunks(segment_c, labels.shape[0], max_dur)
                
                
            target_labels = torch.stack([one_seg[0] for one_seg in segment_c]).unsqueeze(0) + 2 # two is because we are adding our sos and eos
            
            target_durations_unnormalized = self.compute_offsets([one_seg[2] for one_seg in segment_c]).to(target_labels.device).unsqueeze(0)
            segments_dict.append({'seg_gt': target_labels,
                            'seg_dur': target_durations_unnormalized,
                            'seg_dur_normalized': target_durations_unnormalized/target_durations_unnormalized.sum().item(),
                            })
        return segments_dict
        
        
    def convert_labels_to_segments2(self, labels): # , split_segments=False, split_segments_max_dur=None
        segments = self.convert_labels(labels); segments_dict=[]
        for c in range(len(segments)):
            segment_c = segments[c]
            # we need to insert <sos> and <eos>
            segment_c.insert(0, (torch.tensor(-2, device=labels.device), -1, -1))
            segment_c.append((torch.tensor(-1, device=labels.device), segment_c[-1][-1], segment_c[-1][-1]))
            
            target_labels = torch.stack([one_seg[0] for one_seg in segment_c]).unsqueeze(0) + 2 # two is because we are adding our sos and eos
            
            target_durations_unnormalized = self.compute_offsets([one_seg[2] for one_seg in segment_c]).to(target_labels.device).unsqueeze(0)
            segments_dict.append({'seg_gt': target_labels,
                            'seg_dur': target_durations_unnormalized,
                            'seg_dur_normalized': target_durations_unnormalized/target_durations_unnormalized.sum().item(),
                            })
        return segments_dict

    def convert_labels(self,labels):
        label_start_end = []
        for c in range(labels.shape[1]):
            action_borders = []; class_labels = labels[:,c]
            for i in range(len(labels) - 1):
                if class_labels[i] != class_labels[i + 1]:
                    action_borders.append(i)

            action_borders.insert(0, -1)
            action_borders.append(len(labels) - 1)
            class_segments = []
            for i in range(1, len(action_borders)):
                label, start, end = class_labels[action_borders[i]], action_borders[i - 1] + 1, action_borders[i]
                class_segments.append((label, start, end))
            label_start_end.append(class_segments)
        return label_start_end

    def split_segments_into_chunks(self,segments, video_length, max_dur):
        target_durations_unnormalized = self.compute_offsets([one_seg[2] for one_seg in segments]).unsqueeze(0)

        new_segments = []
        for segment, norm_dur in zip(segments, target_durations_unnormalized[0, :] / video_length):
            if norm_dur < max_dur:
                new_segments.append(segment)
            else:
                num_chunks = int(norm_dur.item() // max_dur) + 1
                chunks = np.linspace(segment[1], segment[2] - 1, num=num_chunks + 1, dtype=int)
                start, end = chunks[:-1], chunks[1:] + 1
                for i in range(num_chunks):
                    new_segments.append((segment[0], start[i], end[i]))
        return new_segments

    def compute_offsets(seldf, time_stamps):
        time_stamps.insert(0, -1)
        time_stamps_unnormalized = torch.tensor([float(i - j) for i, j in zip(time_stamps[1:], time_stamps[:-1])])
        return time_stamps_unnormalized

if __name__ == '__main__':
    pass