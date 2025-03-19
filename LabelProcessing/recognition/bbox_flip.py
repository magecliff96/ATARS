import os
import json

def flip_bbox(data):
    resolution = 224
    # Iterate through each video in the data
    for video in data:
        # Iterate through each frame in the video
        for frame in video:
            # Access the bounding box and apply the swap transformations
            bbox = frame['bbox']
            
            # Perform the swap as per the description:
            # [x_min, y_min, x_max, y_max] -> [x_max, y_min, x_min, y_max]
            x_min, y_min, x_max, y_max = bbox
            frame['bbox'] = [resolution-x_max, y_min, resolution-x_min, y_max]
    
    return data

def process_bbox_files(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):  # Process only JSON files
            input_file_path = os.path.join(input_folder, filename)
            
            # Open and read the JSON file
            with open(input_file_path, 'r') as f:
                data = json.load(f)
            
            # Apply the 90-degree rotation to the bounding boxes
            rotated_data = flip_bbox(data)
            
            # Create the output file name by appending "_90" to the original filename
            output_file_name = os.path.splitext(filename)[0] + f'_f.json'
            output_file_path = os.path.join(output_folder, output_file_name)
            
            # Save the rotated data to the new JSON file
            with open(output_file_path, 'w') as f:
                json.dump(rotated_data, f, indent=4)
            
            print(f"Processed {filename}, saved as {output_file_name}")

# Example usage
input_folder = 'jsons2'  # Folder containing original bbox JSON files
output_folder = 'jsons2_f'  # Folder to save modified bbox files

process_bbox_files(input_folder, output_folder)
