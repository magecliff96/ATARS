import os
import numpy as np

# Path to your folder with .npy files
folder_path = r"/home/magecliff/Traffic_Recognition/Carom_TempSeg/features/imgnet"


# Loop through files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.npy'):
        file_path = os.path.join(folder_path, file_name)
        
        # Load the .npy file
        data = np.load(file_path)
        
        # Check if the shape matches (1, 2048, _, 1, 1)
        if not (len(data.shape) == 5 and data.shape[0] == 1 and data.shape[1] == 2048 and data.shape[3] == 1 and data.shape[4] == 1):
            # Reshape if the second dimension is 2048
            if data.shape[1] == 2048:
                target_shape = (1, 2048, data.shape[0], 1, 1)
                
                # Reshape the data to (1, 2048, _, 1, 1)
                reshaped_data = data.transpose(1, 0, 2, 3, 4).reshape(target_shape)
                
                # Save the reshaped data
                np.save(file_path, reshaped_data)
                print(f"Reshaped and saved: {file_name}")
            else:
                print(f"File {file_name} has an unexpected shape: {data.shape}. Skipping.")
        else:
            print(f"File {file_name} already has the correct shape.")