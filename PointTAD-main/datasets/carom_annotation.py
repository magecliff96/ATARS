import os
import pandas as pd
import json

def convert_annotations_to_json(folder_path, frame_rate, output_path, training_folder, testing_folder, data):
    """
    Convert a folder of annotation CSV files into the desired JSON format with subsets assigned.

    Parameters:
        folder_path (str): Path to the folder containing CSV files.
        frame_rate (float): Frame rate of the videos.
        output_path (str): Path to save the output JSON file.
        training_folder (str): Path to the folder containing training file names.
        testing_folder (str): Path to the folder containing testing file names.
        data (list): List of pattern names used to map actions to indices.

    Returns:
        None
    """
    # Get lists of training and testing files
    training_files = {os.path.splitext(f)[0] for f in os.listdir(training_folder) if f.endswith('.MP4')}
    testing_files = {os.path.splitext(f)[0] for f in os.listdir(testing_folder) if f.endswith('.MP4')}

    annotations = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            # Read the CSV file
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)

            # Extract video ID (assume file name without extension)
            video_id = os.path.splitext(file_name)[0]

            # Determine subset
            if video_id in training_files:
                subset = "training"
            elif video_id in testing_files:
                subset = "testing"
            else:
                continue  # Skip files not in training or testing lists

            # Process each action
            actions = []
            for action in df.columns[1:]:  # Skip the 'frame' column
                if action not in data:
                    print(f"Warning: Action '{action}' not found in data list. Skipping.")
                    continue
                
                action_index = data.index(action)+1
                is_active = df[action] == 1
                start_frame = None

                for frame in range(len(is_active)):
                    if is_active[frame]:
                        if start_frame is None:
                            start_frame = frame
                    elif start_frame is not None:
                        # End of a segment
                        start_time = start_frame / frame_rate
                        end_time = frame / frame_rate
                        actions.append([action_index, start_time, end_time])
                        start_frame = None

                # Handle the last segment
                if start_frame is not None:
                    start_time = start_frame / frame_rate
                    end_time = len(is_active) / frame_rate
                    actions.append([action_index, start_time, end_time])

            # Add to annotations
            annotations[video_id] = {
                "subset": subset,
                "duration": len(df) / frame_rate,
                "actions": actions
            }

    # Save to JSON
    with open(output_path, 'w') as json_file:
        json.dump(annotations, json_file, indent=4)

# Example usage
folder_path = r"/home/magecliff/Traffic_Recognition/Carom_TempSeg/csvlabels2"
frame_rate = 30.0  # Example frame rate
training_folder = r"/home/magecliff/Traffic_Recognition/PointTAD-main/data/carom_videos/training"
testing_folder = r"/home/magecliff/Traffic_Recognition/PointTAD-main/data/carom_videos/testing"
output_path = r"/home/magecliff/Traffic_Recognition/PointTAD-main/datasets/carom.json"

# data = ['12v', '12v+', '13v', '13v+', '14v', '14v+', '21v', '21v+', '23v', '23v+', '24v', '24v+', 
#         '31v', '31v+', '32v', '32v+', '34v', '34v+', '41v', '41v+', '42v', '42v+', '43v', '43v+',
#         '12p', '12p+', '14p', '14p+', '21p', '21p+', '23p', '23p+', '32p', '32p+', '34p', '34p+', 
#         '41p', '41p+', '43p', '43p+']
data = ['12v', '12v+', '13v', '13v+', '14v', '14v+', '21v', '21v+', '23v', '23v+', '24v', '24v+', 
        '31v', '31v+', '32v', '32v+', '34v', '34v+', '41v', '41v+', '42v', '42v+', '43v', '43v+',
        '12p', '14p', '21p', '23p', '32p', '34p', '41p', '43p']
convert_annotations_to_json(folder_path, frame_rate, output_path, training_folder, testing_folder, data)
