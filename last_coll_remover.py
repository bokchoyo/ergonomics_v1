import csv

def remove_last_column(input_file, output_file):
    with open(input_file, 'r', newline='') as csv_in:
        reader = csv.reader(csv_in)
        rows = [row[:-1] for row in reader]

    with open(output_file, 'w', newline='') as csv_out:
        writer = csv.writer(csv_out)
        writer.writerows(rows)

# Usage example
input_file = r'C:\Users\bokch\PyCharm\Ergonomics\data\combo_test_labeled.csv'  # Replace with your input file path
output_file = r'C:\Users\bokch\PyCharm\Ergonomics\data\combo_test_labeled.csv'  # Replace with your output file path
remove_last_column(input_file, output_file)
