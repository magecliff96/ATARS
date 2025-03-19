import json
import os

def compare_json_files(file1, file2):
    # Check if both files exist
    if not os.path.exists(file1):
        print(f"File {file1} does not exist.")
        return
    if not os.path.exists(file2):
        print(f"File {file2} does not exist.")
        return

    # Load the content of both JSON files
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)
    
    # Compare the two datasets
    if data1 == data2:
        print(f"Files {file1} and {file2} are the same.")
    else:
        print(f"Files {file1} and {file2} are different.")
        
        # Optionally, you can print the specific differences (if they are small)
        compare_json_content(data1, data2)

def compare_json_content(data1, data2):
    """Compares two JSON data objects and reports differences."""
    if len(data1) != len(data2):
        print(f"File lengths are different: {len(data1)} vs {len(data2)}")

    for i, (frame1, frame2) in enumerate(zip(data1, data2)):
        if frame1 != frame2:
            print(f"Difference found in video at index {i}:")
            print(f"File 1: {frame1}")
            print(f"File 2: {frame2}")
            break  # Stop after finding the first difference (optional)

# Example usage
file1 = r'D:\research\traffic\Traffic_Recognition\Carom2\jsons_rt360\1000_0_0_0_360.json'
file2 = r'D:\research\traffic\Traffic_Recognition\Carom2\subsampled_jsons\1000_0_0_0.json'

compare_json_files(file1, file2)