import csv


def append_columns(csv1_file, csv2_file, output_file):
    with open(csv1_file, 'r', newline='') as csv1, \
            open(csv2_file, 'r', newline='') as csv2, \
            open(output_file, 'w', newline='') as output:
        reader1 = csv.reader(csv1)
        reader2 = csv.reader(csv2)
        writer = csv.writer(output)

        for row1, row2 in zip(reader1, reader2):
            # Append all columns except the first one from csv1 to csv2
            appended_row = row2 + row1[1:]
            writer.writerow(appended_row)

    print(f"Appended data from '{csv1_file}' to '{csv2_file}' and saved to '{output_file}'.")


# Example usage:
csv1_file = r"C:\Users\bokch\PyCharm\Ergonomics\data\side\bad\1001_angles.csv"  # Replace with your first CSV file
csv2_file = r"C:\Users\bokch\PyCharm\Ergonomics\data\front\bad\1001_distances.csv"  # Replace with your second CSV file
output_file = r"C:\Users\bokch\PyCharm\Ergonomics\data\front\bad\1001_training.csv"  # Replace with the output CSV file

append_columns(csv1_file, csv2_file, output_file)
