import os
import json

def remove_extra_brackets(data):
    cleaned_data = []
    for item in data:
        if isinstance(item, list) and len(item) == 1 and isinstance(item[0], list):
            cleaned_data.append(item[0])  # Remove the unnecessary middle bracket
        else:
            cleaned_data.append(item)
    return cleaned_data

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create output folder if it doesn't exist
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):  # Assuming the files are in JSON format
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            
            with open(input_file_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    cleaned_data = remove_extra_brackets(data)
                    
                    # Process frame numbers and class_id conversion
                    processed_data = []
                    frame_id = 0
                    for frame_data in cleaned_data:
                        if isinstance(frame_data, list):
                            frame_id += 1  # Assign sequential frame ID
                            frame_items = []
                            for obj in frame_data:
                                if isinstance(obj, dict):
                                    obj["frame_id"] = frame_id
                                    obj["class_id"] = 1  # Convert class_id to 1
                                    frame_items.append({
                                        "frame_id": obj["frame_id"],
                                        "class_id": obj["class_id"],
                                        "bbox": obj["bbox"]
                                    })
                            processed_data.append(frame_items)
                    
                    with open(output_file_path, "w", encoding="utf-8") as output_file:
                        json.dump(processed_data, output_file, indent=4)
                    print(f"Processed: {filename}")
                except json.JSONDecodeError:
                    print(f"Error reading {filename}: Invalid JSON format")

# Example usage
input_folder = "jsons2"
output_folder = "jsons2_o"
process_folder(input_folder, output_folder)


