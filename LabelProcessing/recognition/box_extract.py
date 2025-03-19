import numpy as np

# Assuming you load the CSV file with the scene data, for example:
data = np.loadtxt("scene_track_data.csv", delimiter=",")
frame_data = []  # This will store data for the whole video

# Loop through each frame
for frame_id in np.unique(data[:, 0]):  # Assuming column 0 is 'export_image_ID' (Frame ID)
    frame_vehicles = data[data[:, 0] == frame_id]  # Get all vehicles in this frame
    frame_list = []  # This will store data for all cars in this frame
    
    # Loop through each vehicle in the frame
    for vehicle in frame_vehicles:
        # Extract relevant columns from vehicle data
        vehicle_id = vehicle[3]  # Assuming column 3 is 'vehicle_ID'
        class_id = vehicle[4]    # Assuming column 4 is 'vehicle_type'
        x_center, y_center = vehicle[21], vehicle[22]  # Position in the XOY plane (columns 21, 22)
        length, width = vehicle[15], vehicle[16]  # Vehicle dimensions (columns 15, 16)

        # Compute bounding box
        x_min = x_center - (length / 2)
        x_max = x_center + (length / 2)
        y_min = y_center - (width / 2)
        y_max = y_center + (width / 2)

        # Create the bounding box dictionary
        vehicle_data = {
            "bbox": [x_min, y_min, x_max, y_max],
            "class_id": class_id
        }
        frame_list.append(vehicle_data)  # Add this vehicle's data to the frame list
    
    # Append this frame's vehicle data to the whole video list
    frame_data.append(frame_list)

# Print or save the output structure in the desired format
import json
print(json.dumps(frame_data, indent=4))
