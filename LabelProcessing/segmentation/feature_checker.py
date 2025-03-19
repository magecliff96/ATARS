import os
import numpy as np

# Folder containing the saved features
feature_folder = r"/home/magecliff/Traffic_Recognition/Carom_TempSeg/features"

# Iterate over the .npy files in the feature folder
for feature_file in os.listdir(feature_folder):
    if feature_file.endswith(".npy"):  # Check if the file is an npy file
        feature_path = os.path.join(feature_folder, feature_file)
        
        # Load the npy file
        features = np.load(feature_path)
        
        # Print the name of the file and its shape
        print(f"Feature file: {feature_file}, Shape: {features.shape}")