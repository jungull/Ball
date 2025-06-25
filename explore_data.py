import pandas as pd

# --- Configuration ---
# This is the file we created in the last step
INPUT_FILE_NAME = "nba_games_raw.csv"

# --- Main script ---
print(f"Loading data from '{INPUT_FILE_NAME}'...")

# Load the csv file into a pandas DataFrame
# A DataFrame is like a powerful, programmable spreadsheet
try:
    df = pd.read_csv(INPUT_FILE_NAME)
    print("Data loaded successfully!")

    # 1. Print the "shape" of the data
    # This tells us (number of rows, number of columns)
    print("\n--- 1. Data Shape ---")
    print(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    # 2. Print the column names and their data types
    print("\n--- 2. Data Types and Non-Null Counts ---")
    df.info()

    # 3. Print a summary of missing values
    print("\n--- 3. Missing Values ---")
    # isnull() creates a table of True/False, sum() counts the Trues
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    if missing_values.sum() == 0:
        print("No missing values found. That's great!")

    # 4. Print descriptive statistics for numerical columns
    # This shows mean, standard deviation, min, max, etc.
    print("\n--- 4. Descriptive Statistics ---")
    # The .T transposes the output to make it easier to read
    print(df.describe().T)

except FileNotFoundError:
    print(f"ERROR: The file '{INPUT_FILE_NAME}' was not found.")
    print("Please run the 'data_collection.py' script first.")
except Exception as e:
    print(f"An error occurred: {e}")