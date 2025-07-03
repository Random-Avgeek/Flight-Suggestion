import pandas as pd

def clean_flight_schedule(input_file_name="Flight_Schedule.csv", output_file_name="fdata_cleaned.csv"):
    """
    Cleans the flight schedule CSV file by:
    1. Deleting rows with any empty fields.
    2. Dropping 'validto' and 'validfrom' columns.

    Args:
        input_file_name (str): The name of the input CSV file.
        output_file_name (str): The name of the output CSV file.
    """
    try:
        df = pd.read_csv(input_file_name)
        print(f"Original DataFrame shape: {df.shape}")
        print("Original DataFrame head:")
        print(df.head())
        df_cleaned_rows = df.dropna()
        print(f"\nDataFrame shape after dropping rows with empty fields: {df_cleaned_rows.shape}")
        columns_to_drop = ['validto', 'validfrom']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df_cleaned_rows.columns]

        if existing_columns_to_drop:
            df_final = df_cleaned_rows.drop(columns=existing_columns_to_drop, axis=1)
            print(f"Columns dropped: {existing_columns_to_drop}")
        else:
            df_final = df_cleaned_rows
            print("No specified columns to drop were found in the DataFrame.")

        print(f"Final DataFrame shape: {df_final.shape}")
        print("Final DataFrame head:")
        print(df_final.head())
        df_final.to_csv(output_file_name, index=False)
        print(f"\nCleaned data successfully saved to '{output_file_name}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_file_name}' was not found in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
if __name__ == "__main__":
    clean_flight_schedule(output_file_name="fdata_cleaned.csv")
