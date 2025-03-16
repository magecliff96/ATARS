#!/usr/bin/python2.7

import torch
import numpy as np
import random


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            num_frames = min(np.shape(features)[1], len(content))
            classes = np.zeros((num_frames, self.num_classes))
            for i in range(num_frames):
                labels_list = list(map(int, content[i].split()))
                classes[i] = labels_list
            batch_input.append(features[:, :num_frames:self.sample_rate])
            batch_target.append(classes[:num_frames:self.sample_rate])

        length_of_sequences = [len(tgt) for tgt in batch_target]
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            seq_len = np.shape(batch_input[i])[1]
            batch_input_tensor[i, :, :seq_len] = torch.from_numpy(batch_input[i])
            tgt_len = np.shape(batch_target[i])[0]
            batch_target_tensor[i, :, :tgt_len] = torch.from_numpy(batch_target[i].T)
            mask[i, :, :tgt_len] = 1

        return batch_input_tensor, batch_target_tensor, mask
