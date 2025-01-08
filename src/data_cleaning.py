import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

def fill_NaNs(df, column_name):
    """
    Fills missing (NaN) values in a specified column with the average value of that column.

    Args:
        df (pd.DataFrame): The original DataFrame.
        column_name (str): The name of the column where NaNs should be filled.

    Returns:
        pd.DataFrame: The updated DataFrame with NaNs in the specified column replaced by the average value.
    """
    # Check if the column exists in the DataFrame
    if column_name in df.columns:
        # Calculate the mean of the column, ignoring NaNs
        mean_value = df[column_name].mean()
        
        # Fill NaN values in the column with the mean value
        df[column_name].fillna(mean_value, inplace=True)
    else:
        print(f"Column '{column_name}' not found in the DataFrame.")
    
    # Return the updated DataFrame
    return df

def check_missing_electrode_values(df):
    # Extract only the electrode columns (assuming they follow a certain pattern)
    electrode_columns = [col for col in df.columns if col not in ['no.', 'sex', 'age', 'eeg.date', 'education', 'IQ', 'main.disorder', 'specific.disorder']]
    
    # Check if there are any missing values in the electrode columns
    missing_values = df[electrode_columns].isna().sum()
    
    # Print the result
    if missing_values.any():
        print("Missing values in the following electrode columns:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values in any of the electrode columns.")

    # Example usage:
    # check_missing_electrode_values(df)

def standardize_categorical_columns(df):
    # List of columns that are categorical (object type)
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Standardize each categorical column (strip spaces and convert to lowercase)
    for col in categorical_columns:
        df[col] = df[col].str.strip().str.lower()
    
    return df

import numpy as np

def check_for_categorical_outliers(df):
    found_outliers = False  # Flag to track if any rare categories are found

    # Iterate through each column in the dataframe
    for col in df.columns:
        # Check for categorical columns (exclude electrode columns)
        if df[col].dtype == 'object' and col not in ['no.', 'sex', 'age', 'eeg.date', 'education', 'IQ', 'main.disorder', 'specific.disorder']:
            # Find categories with low frequency (rare categories)
            category_counts = df[col].value_counts()
            rare_categories = category_counts[category_counts < 5]  # You can adjust the threshold
            
            if not rare_categories.empty:
                found_outliers = True
                print(f"Rare categories detected in categorical column '{col}':")
                print(rare_categories)
                print()
    
    # If no rare categories were found, print this message
    if not found_outliers:
        print("\n No rare categories detected in categorical columns.")

    # Example usage:
    # check_for_categorical_outliers(df)











