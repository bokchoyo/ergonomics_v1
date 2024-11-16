import csv


def combine_csvs_side_by_side(file1, file2, output_file):
    with open(file1, 'r') as csvfile1, open(file2, 'r') as csvfile2, open(output_file, 'w', newline='') as outfile:
        reader1 = csv.reader(csvfile1)
        reader2 = csv.reader(csvfile2)
        writer = csv.writer(outfile)

        # Read all rows from the first CSV
        rows1 = list(reader1)

        # Read every other row from the second CSV
        rows2 = [row for idx, row in enumerate(reader2) if idx % 2 == 0]

        # Calculate the maximum number of rows to combine
        max_rows = max(len(rows1), len(rows2))

        # Combine the rows side by side
        for i in range(max_rows):
            row1 = rows1[i] if i < len(rows1) else []
            row2 = rows2[i] if i < len(rows2) else []
            # Combine the rows, filling with empty strings if rows are of different lengths
            combined_row = row1 + row2
            writer.writerow(combined_row)


# Example usage
combine_csvs_side_by_side('data/landmarks_68.csv', 'data/landmarks_13.csv', 'data/landmarks_81.csv')
