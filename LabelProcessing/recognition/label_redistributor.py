import pandas as pd

# Load the original files without headers
train_data = pd.read_csv('old_train.csv', header=None)
val_data = pd.read_csv('old_val.csv', header=None)

# Combine the train and val data
combined_data = pd.concat([train_data, val_data], ignore_index=True)

# Filter data to get items where the video name starts with "B5" for val, "B3" for new_val_B3, and others for train
test_filtered = combined_data[combined_data[0].str.startswith("B5")]
val_filtered = combined_data[combined_data[0].str.startswith("B3")]
train_filtered = combined_data[~combined_data[0].str.startswith(("B5", "B3"))]

# Save the filtered data into new train, val, and new_val_B3 csv files
train_filtered.to_csv('train.csv', index=False, header=False)
test_filtered.to_csv('test.csv', index=False, header=False)
val_filtered.to_csv('val.csv', index=False, header=False)

print("New train, val, and test files have been created.")
