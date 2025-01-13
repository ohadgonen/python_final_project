import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import mne  

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

def visualize_main_psychiatric_disorders(df):
    """
    Visualizes the frequency of main psychiatric disorders in the dataframe as a bar graph.

    Args:
        df (pd.DataFrame): The input DataFrame containing the 'main.disorder' column.

    Returns:
        None
    """
    if 'main.disorder' not in df.columns:
        print("The 'main.disorder' column is not present in the DataFrame.")
        return

    # Count the occurrences of each main disorder
    disorder_counts = df['main.disorder'].value_counts()

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=disorder_counts.index, y=disorder_counts.values, palette='viridis')

    # Customize the plot
    plt.title('Frequency of Main Psychiatric Disorders', fontsize=16)
    plt.xlabel('Main Psychiatric Disorders', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()

    # Show the plot
    plt.show()



def visualize_eeg_headmap(combined_output, frequency_band='delta'):
    # Create the standard 10-20 EEG montage
    montage = mne.channels.make_standard_montage('standard_1020')
    
    # Prepare the data: we take only the first frequency band for visualization (e.g., 'delta')
    eeg_data = combined_output.filter(like=frequency_band).iloc[0].values  # Get the first row for a specific disorder or frequency band
    
    # Create an Info object with 32 channels (standard for 10-20 system)
    # For this, we need the electrode names and the layout (positions of the electrodes).
    ch_names = [f"{frequency_band}.{electrode}" for electrode in montage.ch_names]  # Use the frequency band as part of the name
    
    info = mne.create_info(ch_names=ch_names, ch_types='eeg', sfreq=256)  # Assuming 256 Hz sampling frequency
    
    # Set the montage (electrode positions) for the Info object
    info.set_montage(montage)
    
    # Create an Evoked object from the EEG data
    evoked_data = np.array([eeg_data])  # Make sure to have the data as a 2D array (time x channels)
    evoked = mne.EvokedArray(evoked_data, info)
    
    # Plot the topomap (the head map)
    evoked.plot_topomap(times='auto', ch_type='eeg', show=True)
    


    

