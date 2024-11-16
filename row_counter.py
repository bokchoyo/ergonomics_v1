import pandas as pd


def count_rows_in_csv(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Count the number of rows
        num_rows = len(df)

        # Print the count
        print(f"Number of rows in '{file_path}': {num_rows}")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")


# Example usage:
file_path = r'C:\Users\bokch\PyCharm\Ergonomics\data\final\training_continuous_3-5-9.csv'
count_rows_in_csv(file_path)
