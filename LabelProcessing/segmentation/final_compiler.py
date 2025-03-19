import pandas as pd
import numpy as np

def combine_excel_data(data1_file, data2_file, output_file):
    # List of additional labels to add to each sheet from data_2
    additional_labels = ['12p', '12p+', '14p', '14p+', '21p', '21p+', '23p', '23p+', 
                         '32p', '32p+', '34p', '34p+', '41p', '41p+', '43p', '43p+']

    # Load data from both Excel files (reading all sheets)
    data1_sheets = pd.read_excel(data1_file, sheet_name=None)  # All sheets from data_1
    data2_sheets = pd.read_excel(data2_file, sheet_name=None)  # All sheets from data_2

    # Create an ExcelWriter object to write the combined data
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Process each sheet in data_1
        for sheet_name, data1 in data1_sheets.items():
            # Set the first column (header) to 'frame' if it's not already
            data1.columns.values[0] = 'frame'
            
            # If the sheet exists in both data_1 and data_2, merge the data
            if sheet_name in data2_sheets:
                data2 = data2_sheets[sheet_name]
                data2.columns.values[0] = 'frame'  # Ensure frame column is named correctly
                
                # Filter out the columns in data_2 that match the additional labels
                data2_filtered = data2[['frame'] + [col for col in data2.columns if col in additional_labels]]
                
                # Merge data_1 and filtered data_2 on the 'frame' column
                combined_data = pd.merge(data1, data2_filtered, on='frame', how='outer')
                
                # Fill missing values in new columns (from data_2) with zeros
                combined_data[additional_labels] = combined_data[additional_labels].fillna(0)
                
            else:
                # If the sheet is not in data_2, create new columns for additional labels and fill with zeros
                for label in additional_labels:
                    data1[label] = 0  # Fill missing columns with zeros
                combined_data = data1

            # Write the combined data to the output Excel file
            combined_data.to_excel(writer, sheet_name=sheet_name, index=False)

# File paths
vehicle_file = 'vehicle_.xlsx'  # Replace with your data_1.xlsx file path
pedestrian_file = 'pedestrian_.xlsx'  # Replace with your data_2.xlsx file path
output_file = 'labels.xlsx'  # Replace with the path for the combined output

# Run the combine process
combine_excel_data(vehicle_file, pedestrian_file, output_file)
