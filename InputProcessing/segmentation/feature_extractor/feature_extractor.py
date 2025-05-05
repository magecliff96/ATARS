import torch
import torch.nn as nn
import numpy as np
from pytorch_i3d import InceptionI3d
import os
import cv2

# Function to pad the video frames with blank frames at the start and end
def pad_video_frames(frames, pad_size_front=5, pad_size_back=4):
    blank_frame = np.zeros_like(frames[0])
    padded_frames = [blank_frame] * pad_size_front + frames + [blank_frame] * pad_size_back
    return padded_frames

# Helper function to load video frames in chunks for RGB and Flow processing
def load_video_frames_for_rgb_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))  # Resize for I3D input
        frames.append(frame)
    cap.release()
    frames = pad_video_frames(frames, pad_size_front=5, pad_size_back=4)
    return frames

# Helper function to compute optical flow for 10 frames (5 before, 1 current, 4 after) using CPU
def compute_optical_flow_10_cpu(frames):
    flow_frames = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Resize the flow to match the I3D model input size
        flow = cv2.resize(flow, (224, 224))
        
        # Flow has 2 channels (x and y flow), append to list
        flow_frames.append(flow)
        
        prev_gray = curr_gray

    return np.array(flow_frames)

# Load models for RGB and Flow, and move to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rgb_model = InceptionI3d(400, in_channels=3).to(device)
flow_model = InceptionI3d(400, in_channels=2).to(device)

# Define the paths for the model weights, input, and output folders
rgb_model_weights = r"/home/magecliff/Traffic_Recognition/feature_extractor/models/rgb_imagenet.pt"
flow_model_weights = r"/home/magecliff/Traffic_Recognition/feature_extractor/models/flow_imagenet.pt"
video_folder = r"/home/magecliff/Traffic_Recognition/Carom_TempSeg/videos"  # Input videos folder
output_folder = r"/home/magecliff/Traffic_Recognition/Carom_TempSeg/features"  # Output features folder

# Load the pre-trained weights
rgb_model.load_state_dict(torch.load(rgb_model_weights, map_location=device))
flow_model.load_state_dict(torch.load(flow_model_weights, map_location=device))

# Set models to evaluation mode
rgb_model.eval()
flow_model.eval()

# Extract features for each video using 100 sets of frames at a time
for video_file in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video_file)
    output_file_combined = os.path.join(output_folder, f'{video_file[:-4]}.npy')
    
    # Check if the feature file already exists
    if os.path.exists(output_file_combined):
        print(f'Skipping {video_file}, features already exist.')
        continue
    
    frames = load_video_frames_for_rgb_flow(video_path)
    
    all_rgb_features = []
    all_flow_features = []
    
    # Process consecutive sets of overlapping frames
    batch_size = 150
    num_sets = len(frames) - 9  # Number of valid 9-frame chunks for RGB (same for Flow due to padding)
    
    for i in range(0, num_sets, batch_size):
        rgb_chunks = []
        flow_chunks = []
        
        for j in range(min(batch_size, num_sets - i)):
            # 9 frames for RGB (4 before, 1 current, 4 after)
            rgb_chunk = np.array(frames[i + j:i + j + 9])
            rgb_chunks.append(rgb_chunk)

            # 10 frames for Flow (5 before, 1 current, 4 after)
            flow_chunk = np.array(frames[i + j:i + j + 10])
            flow_chunks.append(flow_chunk)
        
        # Prepare RGB frames for I3D (batch_size, channels, frames, height, width)
        rgb_frames = torch.from_numpy(np.array(rgb_chunks)).permute(0, 4, 1, 2, 3).float().to(device)
        
        # Prepare flow frames (Compute optical flow for 10 frames)
        flow_chunks = [compute_optical_flow_10_cpu(chunk) for chunk in flow_chunks]
        flow_frames = torch.from_numpy(np.array(flow_chunks)).permute(0, 4, 1, 2, 3).float().to(device)

        # Print progress
        print(f'Processing batch {i // batch_size + 1}/{(num_sets + batch_size - 1) // batch_size}')
        print(f'RGB shape: {rgb_frames.shape}, Flow shape: {flow_frames.shape}')

        # Extract features for RGB and Flow using no_grad for efficiency
        with torch.no_grad():
            rgb_features = rgb_model.extract_features(rgb_frames)
            flow_features = flow_model.extract_features(flow_frames)

            # Append individual frame features from the batch to ensure no size mismatch
            for idx in range(rgb_features.shape[0]):
                all_rgb_features.append(rgb_features[idx:idx + 1])  # Append as batch of 1
                all_flow_features.append(flow_features[idx:idx + 1])  # Append as batch of 1
    
    # Concatenate all features along the temporal dimension (depth axis)
    rgb_features_final = torch.cat(all_rgb_features, dim=0)  # Concatenate along the batch dimension
    flow_features_final = torch.cat(all_flow_features, dim=0)  # Concatenate along the batch dimension
    
    # Concatenate RGB and Flow features along the channel dimension
    combined_features = torch.cat((rgb_features_final, flow_features_final), dim=1)

    # Save concatenated features to the output folder
    np.save(output_file_combined, combined_features.cpu().numpy())
    print(f'Saved features for {video_file} to {output_file_combined}')




# if __name__ == '__main__':
#     video_folder = r"/home/magecliff/Traffic_Recognition/Carom_TempSeg/videos"  # Path to folder with videos
#     dest_folder = r"/home/magecliff/Traffic_Recognition/Carom_TempSeg/features"  # Path to save the extracted features
#     rgb_weights = r"/home/magecliff/Traffic_Recognition/feature_extractor/pytorch-i3d-master/models/rgb_charades.pt"  # Path to RGB pretrained weights
#     flow_weights = r"/home/magecliff/Traffic_Recognition/feature_extractor/pytorch-i3d-master/models/flow_charades.pt"  # Path to Optical Flow pretrained weights

#     # Process all videos in the folder
#     process_videos_in_folder(video_folder, dest_folder, rgb_weights, flow_weights, num_frames=16)
