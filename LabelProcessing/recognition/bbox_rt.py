import os
import json

def rotate_bbox_90deg(data):
    resolution = 224
    # Iterate through each video in the data
    for video in data:
        # Iterate through each frame in the video
        for frame in video:
            # Access the bounding box and apply the swap transformations
            bbox = frame['bbox']
            
            # Perform the swap as per the description:
            # [x_min, y_min, x_max, y_max]
            # x_min => resolution - y_max
            # y_min => x_min
            # x_max => resolution - y_min
            # y_max => x_max
            x_min, y_min, x_max, y_max = bbox
            frame['bbox'] = [resolution - y_max, x_min, resolution - y_min, x_max]
    
    return data

def process_bbox_files(input_folder, output_folder, degree):
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
            rotated_data = rotate_bbox_90deg(data)
            
            # Create the output file name by appending "_90" to the original filename
            if degree == 90:
                output_file_name = os.path.splitext(filename)[0] + f'_{degree}.json'
            else:
                output_file_name = filename.replace(f'_{degree-90}.json', f'_{degree}.json')
            output_file_path = os.path.join(output_folder, output_file_name)
            
            # Save the rotated data to the new JSON file
            with open(output_file_path, 'w') as f:
                json.dump(rotated_data, f, indent=4)
            
            print(f"Processed {filename}, saved as {output_file_name}")

# Example usage
def initiate(degree):
    if degree == 90:
        input_folder = 'jsons2'
    else:
        input_folder = f'jsons2_rt{degree-90}'  # Folder containing original bbox JSON files
    output_folder = f'jsons2_rt{degree}'  # Folder to save modified bbox files

    process_bbox_files(input_folder, output_folder, degree)

initiate(90)
initiate(180)
initiate(270)