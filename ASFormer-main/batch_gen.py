'''
    Adapted from https://github.com/yabufarha/ms-tcn
'''

import torch
import numpy as np
import random
from grid_sampler import GridSampler, TimeWarpLayer

class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

        self.timewarp_layer = TimeWarpLayer()

    def reset(self):
        self.index = 0
        self.my_shuffle()

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        self.list_of_labels = [filename.replace('.MP4', '.txt') for filename in self.list_of_examples]
        self.gts = [self.gt_path + vid for vid in self.list_of_labels]
        self.features = [self.features_path + vid.split('.')[0] + '.npy' for vid in self.list_of_examples]
        self.my_shuffle()

    def my_shuffle(self):
        # shuffle list_of_examples, gts, features with the same order
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(self.list_of_examples)
        random.seed(randnum)
        random.shuffle(self.gts)
        random.seed(randnum)
        random.shuffle(self.features)


    def warp_video(self, batch_input_tensor, batch_target_tensor):
        '''
        :param batch_input_tensor: (bs, C_in, L_in)
        :param batch_target_tensor: (bs, L_in)
        :return: warped input and target
        '''
        bs, _, T = batch_input_tensor.shape
        grid_sampler = GridSampler(T)
        grid = grid_sampler.sample(bs)
        grid = torch.from_numpy(grid).float()

        warped_batch_input_tensor = self.timewarp_layer(batch_input_tensor, grid, mode='bilinear')
        batch_target_tensor = batch_target_tensor.unsqueeze(1).float()
        warped_batch_target_tensor = self.timewarp_layer(batch_target_tensor, grid, mode='nearest')  # no bilinear for label!
        warped_batch_target_tensor = warped_batch_target_tensor.squeeze(1).long()  # obtain the same shape

        return warped_batch_input_tensor, warped_batch_target_tensor

    def merge(self, bg, suffix):
        '''
        merge two batch generator. I.E
        BatchGenerator a;
        BatchGenerator b;
        a.merge(b, suffix='@1')
        :param bg:
        :param suffix: identify the video
        :return:
        '''

        self.list_of_examples += [vid + suffix for vid in bg.list_of_examples]
        self.gts += bg.gts
        self.features += bg.features

        print('Merge! Dataset length:{}'.format(len(self.list_of_examples)))


    def next_batch(self, batch_size, if_warp=False): # if_warp=True is a strong data augmentation. See grid_sampler.py for details.
        batch = self.list_of_examples[self.index:self.index + batch_size]
        batch_gts = self.gts[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size]

        self.index += batch_size

        batch_input = []
        batch_target = []
        for idx, vid in enumerate(batch):
            features = np.load(batch_features[idx])
            file_ptr = open(batch_gts[idx], 'r')
            content_list = file_ptr.read().split('\n')[:-1]

            classes = np.zeros((min(np.shape(features)[2], len(content_list)), len(self.actions_dict)))  # Adjust for video length
            content = [item.split() for item in content_list]
            for i in range(len(classes)):
                for j in range(len(content[i])):
                    index = self.actions_dict[content[i][j]]
                    classes[i][index] = 1
            
            batch_input.append(features)
            batch_target.append(classes)


        total_frames = [min(batch_target[i].shape[0], batch_input[i].shape[1]) for i in range(len(batch_input))]
        for i in range(len(batch_input)):
            batch_input[i] = batch_input[i][:, :, :total_frames[i]]  # Slice batch_input to the minimum temporal dimension
            batch_target[i] = batch_target[i][:total_frames[i]] 


        number_of_frames = list(map(len, batch_target))
        max_length_of_frame = max([len(inner_list) for outer_list in batch_target for inner_list in outer_list])

        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[1], max(number_of_frames), dtype=torch.float)  # bs, C_in, L_in
        batch_target_tensor = torch.ones(len(batch_input), max(number_of_frames), max_length_of_frame, dtype=torch.float32) * (-100)

        mask = torch.zeros(len(batch_input), self.num_classes, max(number_of_frames), dtype=torch.float)
        
        
        for i in range(len(batch_input)):
            batch_input[i] = batch_input[i].squeeze(-1).squeeze(-1)
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0], :np.shape(batch_target[i])[1]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask, batch


if __name__ == '__main__':
    pass