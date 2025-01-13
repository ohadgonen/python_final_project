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
