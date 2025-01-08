import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_non_eeg_data(df):
    # Provides an overview of the dataset, showing only the first 8 columns of interest,excluding all other columns (e.g., electrode columns).
   
    # Columns to keep (the first 8 columns of interest)
    columns_of_interest = [
        'no.', 'sex', 'age', 'eeg.date', 'education', 'IQ', 'main.disorder', 'specific.disorder'
    ]
    
    # Filter the DataFrame to include only the columns of interest
    subset_df = df[columns_of_interest]
    
    # Display concise information about the filtered DataFrame
    print("\n Dataset Overview (Categorical Columns Only)")
    print("-" * 50)
    print(f"Categorical Data shape: {subset_df.shape}\n")
    
    print("Column Names and Data Types:")
    print(subset_df.dtypes)
    print("\nMissing Values:")
    print(subset_df.isnull().sum())
    print("\nDescriptive Statistics:")
    print(subset_df.describe(include="all"))
    
    print("\nFirst 5 Rows of the Filtered DataFrame:")
    print(subset_df.head())

    # Example usage:
    # visualize_data(df)  


def visualize_eeg_data(df):
    # Extract only the electrode columns (assuming they follow a certain pattern)
    # We'll check for columns that do not match the ones defined in the original function (non-electrode columns)
    electrode_columns = [col for col in df.columns if col not in ['no.', 'sex', 'age', 'eeg.date', 'education', 'IQ', 'main.disorder', 'specific.disorder']]
    
    print("\n Dataset Overview (EEG Columns Only)")
    print("-" * 50)

    # Get the shape of the DataFrame
    print(f"EEG Data shape: {df[electrode_columns].shape}")
    
    # Display the first 5 rows of the relevant columns
    print("\nFirst 5 rows (electrode columns only):")
    print(df[electrode_columns].head())
    
    # Show column names and data types
    print("\nColumn Names and Data Types:")
    print(df[electrode_columns].dtypes)
    
    # Count missing values in each column
    print("\nMissing Values in Each Electrode Column:")
    print(df[electrode_columns].isna().sum())
    
    # Display descriptive statistics for the electrode columns
    print("\nDescriptive Statistics (electrode columns only):")
    print(df[electrode_columns].describe())
    # Example usage:
    # visualize_eeg_data(df) 