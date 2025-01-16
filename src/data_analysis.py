import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def data_exploration(df):

    """Explore the DataFrame structure"""

    print("Columns and Data Types:\n", df.dtypes)

    print("\nSummary Statistics:\n", df.describe(include='all'))

    print("\nMissing Values:\n", df.isnull().sum())

    print("\nSample Data:\n", df.head())

from src.data_cleaning import reformat_electrode_columns

def compute_mean_by_disorder(dataframe, frequency_bands):
    # Ensure the electrode columns are reformatted using the provided function
    dataframe = reformat_electrode_columns(dataframe)
    
    # Get all electrode names (assuming they follow the format 'band.channel')
    electrodes = sorted(set([col.split('.')[-1] for col in dataframe.columns if '.' in col]))

    # Create a list of columns for each frequency band and electrode, ensuring correct column names
    result_columns = [f"{band}.{electrode}" for band in frequency_bands for electrode in electrodes]

    # Initialize a DataFrame to store the results with the correct columns
    result = pd.DataFrame(columns=result_columns)

    # Iterate over all the main disorders to compute the mean for each frequency band and each electrode
    for disorder in dataframe['main.disorder'].dropna().unique():  # Handle NaN values
        row_values = {}
        
        # Compute the mean values for each frequency band and each electrode
        for electrode in electrodes:
            for band in frequency_bands:
                electrode_band = f"{band}.{electrode}"
                band_data = [col for col in dataframe.columns if electrode in col and band in col]
                row_values[electrode_band] = dataframe[dataframe['main.disorder'] == disorder][band_data].mean().mean()

        # Add the computed values for this disorder to the result DataFrame
        result.loc[disorder] = row_values
    
    # Return the final DataFrame
    return result

def calculate_band_averages(main_disorder, df):
    """
    Calculate the average values for each electrode in specified frequency bands
    for a given main disorder.
    
    Parameters:
    main_disorder (str): The target main disorder to filter by.
    df (pd.DataFrame): The EEG data.
    
    Returns:
    dict: A dictionary where keys are frequency bands and values are dictionaries
          with electrodes as keys and their averages as values.
    """
    # Frequency bands and their prefixes
    frequency_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'highbeta']
    electrodes = [
        'FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8',
        'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'
    ]
    
    # Filter the dataframe for the specified main disorder
    filtered_df = df[df['main.disorder'] == main_disorder]
    
    result = {}
    for band in frequency_bands:
        # Extract columns matching the band and electrodes
        band_columns = [col for col in df.columns if col.startswith(band) and any(e in col for e in electrodes)]
        
        # Calculate averages for each electrode
        band_averages = {
            col.split('.')[-1]: filtered_df[col].mean() for col in band_columns
        }
        result[band] = band_averages
    
    return result


def prepare_disorder_band_averages(dataframe, disorders, frequency_bands, electrodes):
    """
    Prepare the required arguments for the `visualize_all_disorders` function from the given dataset.

    Parameters:
    dataframe (pd.DataFrame): The EEG dataset containing electrode data and disorder labels.
    disorders (list): List of unique disorders to include (corresponds to `main.disorder` column in the dataset).
    frequency_bands (list): List of frequency bands to extract (e.g., ["delta", "theta", "alpha", "beta", "gamma", "highbeta"]).
    electrodes (list): List of electrode names to include (e.g., ["FP1", "FP2", "F3", ..., "O2"]).

    Returns:
    tuple: A tuple containing:
           - disorder_band_averages: A list of dictionaries (one for each disorder).
           - disorder_names: A list of disorder names.
    """
    disorder_band_averages = []
    disorder_names = []

    for disorder in disorders:
        # Filter the dataset for the specific disorder
        filtered_df = dataframe[dataframe['main.disorder'] == disorder]

        # Initialize the dictionary for the current disorder
        band_averages = {}
        
        for band in frequency_bands:
            # Extract the relevant columns for the frequency band
            band_columns = [f"{band}.{electrode}" for electrode in electrodes if f"{band}.{electrode}" in dataframe.columns]
            
            # Calculate the average for each electrode in the band
            band_averages[band] = {
                electrode: filtered_df[f"{band}.{electrode}"].mean() for electrode in electrodes if f"{band}.{electrode}" in band_columns
            }
        
        # Append the result
        disorder_band_averages.append(band_averages)
        disorder_names.append(disorder)

    return disorder_band_averages, disorder_names


from scipy.stats import ttest_ind
# Re-import necessary libraries after execution reset
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Define the function with filtering for enhanced or lower activity
def find_significant_differences(df, disorder_name, healthy_control_name, p_threshold=0.1, activity_type="both"):
    """
    Identify electrodes with significant differences in activity between a specific disorder and healthy controls.

    Parameters:
    df (pd.DataFrame): The full EEG dataset containing disorder and healthy control data.
    disorder_name (str): The main disorder to compare.
    healthy_control_name (str): The label for the healthy control group.
    p_threshold (float): The significance threshold for the t-test (default is 0.1).
    activity_type (str): "enhanced" for higher activity, "lower" for reduced activity, "both" for all significant differences.

    Returns:
    dict: A dictionary where keys are frequency bands, and values are lists of electrodes with significant differences.
    """
    significant_electrodes = {}

    # Extract healthy control and disorder-specific data
    disorder_df = df[df["main.disorder"] == disorder_name]
    healthy_df = df[df["main.disorder"] == healthy_control_name]

    # Ensure that we have data for both groups
    if disorder_df.empty or healthy_df.empty:
        print(f"Warning: No data found for {disorder_name} or {healthy_control_name}.")
        return significant_electrodes

    # Define frequency bands and electrodes
    frequency_bands = ["delta", "theta", "alpha", "beta", "gamma", "highbeta"]
    electrodes = ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8',
                  'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']

    for band in frequency_bands:
        significant_electrodes[band] = []

        for electrode in electrodes:
            column_name = f"{band}.{electrode}"

            if column_name in df.columns:
                # Extract non-null values for each group
                disorder_values = disorder_df[column_name].dropna().values
                healthy_values = healthy_df[column_name].dropna().values

                # Ensure both groups have enough data to compare
                if len(disorder_values) > 1 and len(healthy_values) > 1:
                    # Perform independent t-test
                    t_stat, p_value = ttest_ind(disorder_values, healthy_values, equal_var=False)

                    if p_value < p_threshold:
                        if activity_type == "enhanced" and disorder_values.mean() > healthy_values.mean():
                            significant_electrodes[band].append(electrode)
                        elif activity_type == "lower" and disorder_values.mean() < healthy_values.mean():
                            significant_electrodes[band].append(electrode)
                        elif activity_type == "both":
                            significant_electrodes[band].append(electrode)

    return significant_electrodes

