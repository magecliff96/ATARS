import pandas as pd
import numpy as np  # Import if necessary for additional operations
import shutil  # Import if necessary for file operations

# Define the path to the input Excel file
file_path = r'D:\research\traffic\label_tool\label\CAROM_Air_Part_2.xlsx'
output_file_path = r'D:\research\traffic\label_tool\label\sampled_intervals2.xlsx'
length = 9
# Define function to sample 3-second intervals while avoiding "move" intervals
def sample_intervals(df, length, fps=30):
    # Skip the first row and filter out non-numeric columns
    numeric_df = df.iloc[1:].apply(pd.to_numeric, errors='coerce')

    # Assuming the max value represents the length of the video in frames
    video_length = numeric_df.max().max()
    interval_length = length * fps  # Convert 3 seconds to frames

    # Collect all "move" intervals and other labels intervals
    move_intervals = []
    other_intervals = []
    for col in df.columns:
        if col.endswith('_s'):
            start_col = col
            end_col = col.replace('_s', '_e')
            if start_col in df.columns and end_col in df.columns:
                for start, end in zip(df[start_col].dropna(), df[end_col].dropna()):
                    if 'move' in col:
                        move_intervals.append((start, end))
                    else:
                        other_intervals.append((start, end))

    # Function to check if an interval overlaps with any "move" interval
    def is_valid_interval(start, end):
        for move_start, move_end in move_intervals:
            if not (end < move_start or start > move_end):
                return False
        return True

    # Function to check if an interval is too close to other label intervals
    def is_not_too_close_to_labels(start, end):
        for label_start, label_end in other_intervals:
            if  label_end >= start >= label_end - 15 or label_start <= end <= label_start + 15:
                return False
        return True

    # Function to check if an interval overlaps with any label interval
    def overlaps_with_any_label(start, end):
        for label_start, label_end in other_intervals:
            if not (end < label_start or start > label_end):
                return True
        return False

    # Sample valid intervals
    sampled_intervals = []
    start = 0
    while start + interval_length <= video_length:
        end = start + interval_length
        if is_valid_interval(start, end) and is_not_too_close_to_labels(start, end) and overlaps_with_any_label(start, end):
            sampled_intervals.append((start, end))
            start = end  # Move to the next interval
        else:
            start += fps  # Move by 1 second if the interval is not valid

    return sampled_intervals

# Load the Excel file
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names

# Process each sheet
sampled_results = {}
for sheet in sheet_names:
    if sheet not in ['summary', 'outline', 'template']:
        print(f"Processing sheet: {sheet}")
        df = pd.read_excel(file_path, sheet_name=sheet)
        intervals = sample_intervals(df,length)
        sampled_results[sheet] = intervals

# Display the sampled intervals for each sheet
for sheet, intervals in sampled_results.items():
    print(f"Sheet: {sheet}")
    for interval in intervals:
        print(f"Start: {interval[0]}, End: {interval[1]}")
    print("\n")

# Save the sampled intervals into separate sheets of an Excel file
with pd.ExcelWriter(output_file_path) as writer:
    for sheet in sheet_names:
        if sheet not in ['summary', 'outline', 'template']:
            clean_sheet_name = sheet.replace('.mp4', '')
            df = pd.read_excel(file_path, sheet_name=sheet)
            intervals = sample_intervals(df,length)
            interval_data = [{'clip_name': f'{clean_sheet_name}_{idx}', 'start_frame': start, 'end_frame': end} for idx, (start, end) in enumerate(intervals)]
            interval_df = pd.DataFrame(interval_data)
            interval_df.to_excel(writer, sheet_name=clean_sheet_name, index=False)

# Provide a download link for the output file
output_file_path