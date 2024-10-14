import pandas as pd

# Load the CSV file into a DataFrame
csv_file_path = 'simulation_results.csv'
df = pd.read_csv(csv_file_path)

# Set display option to show all rows
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(df)