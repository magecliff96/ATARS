import os
import pandas as pd
from glob import glob


def compile_excel_files(folder_path, output_file):
    """Compile multiple Excel files into one, warning if sheet names overlap."""
    # Use glob to get a list of all Excel files in the folder
    excel_files = glob(os.path.join(folder_path, '*.xlsx'))
    
    # Track sheet names to check for overlaps
    all_sheets = set()

    # Create an ExcelWriter object to write all sheets into one file
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for file in excel_files:
            # Load the Excel file
            excel_data = pd.read_excel(file, sheet_name=None)  # None reads all sheets
            
            # Loop through each sheet in the file and write it to the new file
            for sheet_name, sheet_data in excel_data.items():
                # Check for sheet name overlap
                if sheet_name in all_sheets:
                    print(f"Warning: Sheet name '{sheet_name}' from file '{os.path.basename(file)}' already exists.")
                else:
                    all_sheets.add(sheet_name)  # Add the sheet name to the set
                
                # Write the sheet to the new Excel file with its original sheet name
                sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)

# Define the folder path and output file
folder_path = 'vehicle'
output_file = 'vehicle.xlsx'  # Change the path and name as desired

# Call the function to compile Excel files
compile_excel_files(folder_path, output_file)
