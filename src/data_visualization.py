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


from mne.channels import make_standard_montage
from mne.viz import plot_topomap

def visualize_brain_activity(band_averages, main_disorder):
    """
    Visualize brain activity as topographical head maps for each frequency band.
    
    Parameters:
    band_averages (dict): Dictionary with frequency bands as keys and dictionaries
                          of electrode averages as values.
    main_disorder (str): The main disorder being presented.
    
    Returns:
    None: Displays the topographical maps as images.
    """
    # Frequency bands to visualize
    frequency_bands = list(band_averages.keys())
    electrodes = list(band_averages[frequency_bands[0]].keys())
    
    # Electrode positions (standard 10-20 EEG system)
    montage = mne.channels.make_standard_montage('standard_1020')
    positions = montage.get_positions()['ch_pos']
    
    # Convert positions from 3D to 2D
    pos_2d = {ch: positions[ch][:2] for ch in electrodes if ch in positions}
    pos_array = np.array(list(pos_2d.values()))
    
    # Create a figure with enough spacing between subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, band in enumerate(frequency_bands):
        # Prepare data for the current band
        data = np.array([band_averages[band].get(elec, np.nan) for elec in electrodes if elec in pos_2d])
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        
        # Create the topomap with proper arguments
        im, _ = mne.viz.plot_topomap(
            data,
            pos_array,
            axes=axes[idx],
            show=False,
            cmap='viridis',
            sensors=False,
            vlim=(vmin, vmax),
            contours=0,  # Disable contours for a cleaner look
            outlines="head"  # Adjust outline to ensure proper fit
        )
        axes[idx].set_title(f'{band.capitalize()} Band', fontsize=16)
    
    # Add a title for the main disorder
    fig.suptitle(f'Brain Activity for {main_disorder}', fontsize=18, y=0.95)
    
    # Adjust the spacing of the plots
    plt.subplots_adjust(left=0.05, right=0.85, top=0.88, bottom=0.1, wspace=0.4, hspace=0.4)
    
    # Add a colorbar to indicate activity strength
    cbar_ax = fig.add_axes([0.88, 0.2, 0.02, 0.6])  # Position the colorbar on the side
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Activity Strength', fontsize=14)
    
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import mne

def visualize_all_disorders(disorder_band_averages, disorder_names):
    """
    Visualize brain activity as topographical head maps for each frequency band across all disorders.
    
    Parameters:
    disorder_band_averages (list): A list of 7 dictionaries, each corresponding to one disorder.
                                   Each dictionary contains frequency bands as keys and electrode averages as values.
    disorder_names (list): A list of 7 disorder names corresponding to the dictionaries in `disorder_band_averages`.
    
    Returns:
    None: Displays the topographical maps as images.
    """
    # Ensure input sizes match
    if len(disorder_band_averages) != 7 or len(disorder_names) != 7:
        raise ValueError("Both `disorder_band_averages` and `disorder_names` must contain 7 elements.")
    
    # Define frequency bands in order of decreasing Hz
    frequency_bands = ["gamma", "highbeta", "beta", "alpha", "theta", "delta"]

    # Get electrode positions (standard 10-20 EEG system)
    montage = mne.channels.make_standard_montage('standard_1020')
    positions = montage.get_positions()['ch_pos']
    
    # Create a figure for all disorders and frequency bands
    fig, axes = plt.subplots(6, 7, figsize=(24, 18))  # Disorders on x-axis, frequencies on y-axis
    fig.subplots_adjust(left=0.05, right=0.85, top=0.92, bottom=0.08, wspace=0.6, hspace=0.3)  # Increased spacing
    
    for disorder_idx, (band_averages, disorder_name) in enumerate(zip(disorder_band_averages, disorder_names)):
        electrodes = list(band_averages[frequency_bands[0]].keys())
        
        # Convert positions from 3D to 2D
        pos_2d = {ch: positions[ch][:2] for ch in electrodes if ch in positions}
        pos_array = np.array(list(pos_2d.values()))
        
        for band_idx, band in enumerate(frequency_bands):
            # Prepare data for the current band
            data = np.array([band_averages[band].get(elec, np.nan) for elec in electrodes if elec in pos_2d])
            vmin, vmax = np.nanmin(data), np.nanmax(data)
            
            # Get the corresponding axis
            ax = axes[band_idx, disorder_idx]  # Swap indices for horizontal layout
            
            # Create the topomap
            im, _ = mne.viz.plot_topomap(
                data,
                pos_array,
                axes=ax,
                show=False,
                cmap='viridis',
                sensors=False,
                vlim=(vmin, vmax),
                contours=0,
                outlines="head"
            )
            if band_idx == 0:
                ax.set_title(disorder_name, fontsize=14, pad=20)  # Add padding to space titles more
            if disorder_idx == 0:
                ax.set_ylabel(f'{band.capitalize()} Band', fontsize=14)
    
    # Add a global colorbar
    cbar_ax = fig.add_axes([0.88, 0.2, 0.02, 0.6])  # Position the colorbar on the side
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Activity Strength', fontsize=14)
    
    plt.show()
