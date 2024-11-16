import pandas as pd


def keep_top_rows(input_csv, output_csv, num_rows=3500):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Keep only the top 'num_rows' rows
    top_rows = df.head(num_rows)

    # Write the top rows to a new CSV file
    top_rows.to_csv(output_csv, index=False)


# Example usage
input_csv = r'C:\Users\bokch\PyCharm\Ergonomics\data\final\test_0_testing_v2_v2.csv'  # Replace with your input CSV file name
output_csv = r'C:\Users\bokch\PyCharm\Ergonomics\data\final\test_0_testing_reduced.csv'  # Replace with your desired output CSV file name
keep_top_rows(input_csv, output_csv)
