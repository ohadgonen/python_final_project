import numpy as np
import pandas as pd

def evaluate_lobe_activity(df, participant_index, frequency, lobe):
    """
    Evaluates the level of lobe activity for a participant for a specific frequency range.
    
    Parameters:
        df (pd.DataFrame): The EEG dataset.
        participant_index (int): The index of the participant in the dataset.
        frequency (str): The frequency range (e.g., 'delta', 'theta', 'alpha', 'beta', 'gamma').
        lobe (str): The lobe to evaluate ('frontal', 'temporal', 'parietal', 'occipital').
    
    Returns:
        float: The aggregated lobe activity level for the specified frequency range.
    """
    # Define electrode mapping for lobes
    lobe_electrodes = {
        'frontal': ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'FZ'],
        'temporal': ['T3', 'T4', 'T5', 'T6'],
        'parietal': ['P3', 'P4', 'PZ'],
        'occipital': ['O1', 'O2'],
    }
    
    if lobe not in lobe_electrodes:
        raise ValueError(f"Lobe '{lobe}' is not recognized. Valid options are: {list(lobe_electrodes.keys())}.")
    
    # Get the participant row
    participant_data = df.iloc[participant_index]
    
    # Select relevant columns for the specified frequency and lobe
    relevant_columns = [
        col for col in df.columns 
        if any(electrode in col for electrode in lobe_electrodes[lobe]) and frequency in col
    ]
    
    # Sum or average the activity in the relevant columns
    lobe_activity = participant_data[relevant_columns].sum()
    
    return lobe_activity

# Example usage:
# Assuming `eeg_df` is your dataframe
# participant_index = 0 (first participant)
# frequency = 'delta'
# lobe = 'frontal'
# activity = evaluate_lobe_activity(eeg_df, 0, 'delta', 'frontal')
# print(activity)


def data_exploration(df):

    """Explore the DataFrame structure."""

    print("Columns and Data Types:\n", df.dtypes)

    print("\nSummary Statistics:\n", df.describe(include='all'))

    print("\nMissing Values:\n", df.isnull().sum())

    print("\nSample Data:\n", df.head())










