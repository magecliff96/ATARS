import csv

def compare_csv(file1, file2):
    # Open both CSV files
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        
        line_num = 0
        differences_found = False

        # Loop through both files line by line
        for row1, row2 in zip(reader1, reader2):
            line_num += 1

            # Compare the rows
            if row1 != row2:
                differences_found = True
                print(f"Difference found at line {line_num}:")
                print(f"File 1: {row1}")
                print(f"File 2: {row2}")
                print()

        # Check if one file has more lines than the other
        remaining_rows_f1 = list(reader1)
        remaining_rows_f2 = list(reader2)

        if remaining_rows_f1 or remaining_rows_f2:
            differences_found = True
            print(f"One file has more lines than the other:")
            if remaining_rows_f1:
                print(f"File 1 has extra lines starting at {line_num + 1}:")
                for row in remaining_rows_f1:
                    print(f"File 1: {row}")
            if remaining_rows_f2:
                print(f"File 2 has extra lines starting at {line_num + 1}:")
                for row in remaining_rows_f2:
                    print(f"File 2: {row}")

        if not differences_found:
            print("The two files have the exact same data.")

# Call the function with your file paths
compare_csv(r'csvlabels_ff\C1_2_3_f_f.csv', r'csvlabels\C1_2_3.csv')
