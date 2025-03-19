import pandas as pd

# List of desired labels (including the frame number)
desired_labels = ['frame_number', '12p', '12p+', '14p', '14p+', '21p', '21p+', '23p', '23p+', 
                  '32p', '32p+', '34p', '34p+', '41p', '41p+', '43p', '43p+']

def process_excel_file(input_file, output_file):
    # Read the Excel file, including all sheets
    excel_data = pd.read_excel(input_file, sheet_name=None)  # Dictionary with sheet_name as keys
    
    # Create a writer object for the output file
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Process each sheet
        for sheet_name, sheet_data in excel_data.items():
            # Assuming the first column is the frame number
            if 'frame_number' not in sheet_data.columns:
                sheet_data.insert(0, 'frame_number', sheet_data.iloc[:, 0])

            # Filter columns that are in the desired labels (ignore irrelevant columns)
            relevant_columns = [col for col in sheet_data.columns if col in desired_labels]
            filtered_data = sheet_data[relevant_columns]
            
            # Ensure all desired labels are present, filling missing ones with zeros
            for label in desired_labels:
                if label not in filtered_data.columns:
                    filtered_data[label] = 0  # Fill missing columns with zeros
            
            # Reorder the columns to match the desired labels order
            filtered_data = filtered_data[desired_labels]
            
            # Write the processed data to the new sheet in the output file
            filtered_data.to_excel(writer, sheet_name=sheet_name, index=False)

# File paths
input_file = 'pedestrian\Test_CAROM_Air_Pedestrian.xlsx'  # Replace with the path to your input file
output_file = 'pedestrian.xlsx'  # Replace with the path for the output file

# Run the process
process_excel_file(input_file, output_file)
