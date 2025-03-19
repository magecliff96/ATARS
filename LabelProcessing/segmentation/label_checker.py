import pandas as pd
import numpy as np

def process_excel_file(input_file, output_file):
    # Load the Excel file (reading all sheets)
    excel_data = pd.read_excel(input_file, sheet_name=None)  # Read all sheets
    
    # Check if there are exactly 39 sheets
    if len(excel_data) != 29:
        raise ValueError(f"Expected 39 sheets, but found {len(excel_data)} sheets.")

    # Create an ExcelWriter object to write the processed data to a new file
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, sheet_data in excel_data.items():
            # Set the first cell (header of the frame column) to "frame"
            sheet_data.columns.values[0] = 'frame'
            
            # Ensure the frame column starts at 0
            if sheet_data['frame'].iloc[0] != 0:
                # Find the starting frame number
                first_frame = sheet_data['frame'].iloc[0]
                
                # Create rows for missing frames (from 0 up to the first frame)
                missing_frames = pd.DataFrame(np.zeros((first_frame, len(sheet_data.columns))),
                                              columns=sheet_data.columns)
                missing_frames['frame'] = range(0, first_frame)  # Set frame numbers from 0 to first_frame-1
                
                # Concatenate missing frames with the original data
                sheet_data = pd.concat([missing_frames, sheet_data], ignore_index=True)

            # Write the processed data to the output Excel file
            sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)

# Define the input and output file paths
input_file = 'pedestrian.xlsx'  # Replace with the path to your input file
output_file = 'pedestrian_.xlsx'  # Replace with the path for your output file

# Run the process
process_excel_file(input_file, output_file)
