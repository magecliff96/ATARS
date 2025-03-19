import pandas as pd
import os

data_columns = ['12v', '12v+', '13v', '13v+', '14v', '14v+', '21v', '21v+', '23v', '23v+', '24v', '24v+', 
               '31v', '31v+', '32v', '32v+', '34v', '34v+', '41v', '41v+', '42v', '42v+', '43v', '43v+',
               '12p', '12p+', '14p', '14p+', '21p', '21p+', '23p', '23p+', '32p', '32p+', '34p', '34p+', 
               '41p', '41p+', '43p', '43p+']
# Function to switch numbers based on the presence of 'v' or 'p'
def switch_numbers(item):
    # Mapping to replace each number with the new corresponding number
    switch_map = {'2': '4', '4': '2'}
    # Replace each character in the item if it is in the switch_map
    new_item = ''.join(switch_map.get(ch, ch) for ch in item)
    return new_item

# Apply the switch function to each item in the list
switched_data = [switch_numbers(item) for item in data_columns]

# Finding index transformation
index_transformation = {switched_data.index(item): data_columns.index(item) for item in switched_data}

# Define the input and output folders
input_folder = './csvlabels_f'  # Change this to your input folder path
output_folder = './csvlabels_ff'  # Change this to your output folder path
os.makedirs(output_folder, exist_ok=True)


# Process each CSV file in the folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        csv_path = os.path.join(input_folder, filename)
        
        # Load the CSV, keeping the first column (frame number) separate
        data = pd.read_csv(csv_path, header=None)
        
        # Extract and save the first row separately
        first_row = data.iloc[0].copy()

        # Separate the frame number column and other columns
        frame_numbers = data.iloc[:, 0]  # Extract the frame number column
        other_data = data.iloc[:, 1:]    # All columns except the frame number

        # Apply the transformations only to the non-frame columns
        transformed_data = other_data.iloc[:, [index_transformation.get(i, i) for i in range(len(other_data.columns))]]

        # Re-insert the frame number column at the beginning
        transformed_data.insert(0, 0, frame_numbers)

        # Replace the first row of `transformed_data` with the original first row
        transformed_data.iloc[0] = first_row

        # Modify the filename based on `degree`
        output_filename = filename.replace(f'.csv', f'_f.csv')
        
        output_csv_path = os.path.join(output_folder, output_filename)

        # Save the rearranged DataFrame to a new CSV file in the output folder
        transformed_data.to_csv(output_csv_path, index=False, header=None)


        print(f"Processed and saved: {output_csv_path}")