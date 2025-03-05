'''
    Adapted from https://github.com/yabufarha/ms-tcn
'''

import torch
import numpy as np
import random
import cv2

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
        self.features = [self.features_path + vid.split('.')[0] + '.MP4' for vid in self.list_of_examples]
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


    def load_video_file(file_path, sample_rate=1):
        cap = cv2.VideoCapture(file_path)
        frames = []

        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {file_path}")
            return None

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Sample frames according to the sample rate
            if frame_idx % sample_rate == 0:
                # Convert frame to float32 and transpose it to match the expected format (channels, height, width)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (optional, depends on your model's needs)
                frame = frame.astype(np.float32)
                frame = np.transpose(frame, (2, 0, 1))  # Change shape from (H, W, C) to (C, H, W)
                frames.append(frame)
            frame_idx += 1

        cap.release()

        return np.array(frames)  # Return frames as a NumPy array (channels, height, width, time)



    def next_batch(self, batch_size, if_warp=False):  # if_warp=True is a strong data augmentation. See grid_sampler.py for details.
        batch = self.list_of_examples[self.index:self.index + batch_size]
        batch_gts = self.gts[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size]  

        self.index += batch_size

        batch_input = []
        batch_target = []
        for idx, vid in enumerate(batch):
            # Load video frames from the .mp4 file instead of .npy
            features = BatchGenerator.load_video_file(batch_features[idx])
            if features is None:
                continue  # Skip if video couldn't be loaded

            # Load the ground truth classes
            file_ptr = open(batch_gts[idx], 'r')
            content_list = file_ptr.read().split('\n')[:-1]
            file_ptr.close()

            # Create classes array, ensuring it matches the length of the video frames
            classes = np.zeros((min(np.shape(features)[0], len(content_list)), len(self.actions_dict)))  # Adjust for video length
            content = [item.split() for item in content_list]
            for i in range(len(classes)):
                for j in range(len(content[i])):
                    index = self.actions_dict[content[i][j]]
                    classes[i][index] = 1

            # Sample the features and targets according to the sample rate
            feature = features[::self.sample_rate, :, :, :]
            target = classes[::self.sample_rate]

            batch_input.append(feature)
            batch_target.append(target)

        number_of_frames = list(map(len, batch_target))
        max_length_of_frame = max([len(inner_list) for outer_list in batch_target for inner_list in outer_list])

        batch_target_tensor = torch.ones(len(batch_input), max(number_of_frames), max_length_of_frame, dtype=torch.float32) * (-100)
        for i in range(len(batch_input)):
            batch_target_tensor[i, :np.shape(batch_target[i])[0], :np.shape(batch_target[i])[1]] = torch.from_numpy(batch_target[i])
        batch_input_tensor = torch.tensor(batch_input, dtype=torch.float32)

        return batch_input_tensor, batch_target_tensor, batch 

if __name__ == '__main__':
    pass