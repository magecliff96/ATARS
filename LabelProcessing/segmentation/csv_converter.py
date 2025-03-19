import os
import pandas as pd

def convert_xlsx_to_csv(xlsx_file, output_dir=None):
    # Load the Excel file (reading all sheets)
    excel_data = pd.read_excel(xlsx_file, sheet_name=None)  # Read all sheets
    
    # If no output directory is specified, use the same directory as the input file
    if output_dir is None:
        output_dir = os.path.dirname(xlsx_file)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the file name without extension for naming CSV files
    file_name = os.path.splitext(os.path.basename(xlsx_file))[0]
    
    # Convert each sheet to a CSV file
    for sheet_name, sheet_data in excel_data.items():
        # Generate a CSV file name based on the sheet name
        csv_file = os.path.join(output_dir, f"{sheet_name}.csv")
        
        # Save the sheet to a CSV file
        sheet_data.to_csv(csv_file, index=False)
        print(f"Converted '{sheet_name}' to CSV: {csv_file}")

# Define the path to the input Excel file
xlsx_file = 'labeling\labels.xlsx'  # Replace with the path to your .xlsx file

# Optionally define the output directory (leave as None to use the same directory as the input file)
output_dir = r'D:\research\traffic\Traffic_Recognition\Carom_TempSeg\csvlabels'  # You can set a specific path if desired

# Run the conversion
convert_xlsx_to_csv(xlsx_file, output_dir)
