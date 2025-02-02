import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mne  
import networkx as nx

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
    
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_main_psychiatric_disorders(df):
    """
    Visualizes the frequency of main psychiatric disorders in the dataframe as a bar graph.

    Args:
        df (pd.DataFrame): The input DataFrame containing the 'main.disorder' column.

    Returns:
        None
    """
    if 'main.disorder' not in df.columns:
        raise ValueError("The 'main.disorder' column is not present in the DataFrame.")

    if df['main.disorder'].isnull().all():
        raise ValueError("The 'main.disorder' column contains only NaN values.")

    # Count the occurrences of each main disorder
    disorder_counts = df['main.disorder'].value_counts()

    # Ensure there is data to plot
    if disorder_counts.empty:
        print("No data available to plot.")
        return

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=disorder_counts.index, y=disorder_counts.values, hue=disorder_counts.index, palette='viridis', legend=False)
    
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
    """
    if not isinstance(band_averages, dict) or not band_averages:
        raise ValueError("band_averages must be a non-empty dictionary.")
    
    # Remove None values and filter out empty frequency bands
    band_averages = {
        band: {k: v for k, v in values.items() if v is not None}
        for band, values in band_averages.items()
    }
    
    # Ensure at least one valid frequency band remains
    band_averages = {band: values for band, values in band_averages.items() if values}
    if not band_averages:
        raise ValueError("All frequency bands contain only None values or are empty.")
    
    # Frequency bands to visualize
    frequency_bands = list(band_averages.keys())
    electrodes = list(next(iter(band_averages.values())).keys())
    
    if not electrodes:
        raise ValueError("No valid electrodes found in band_averages.")
    
    # Electrode positions (standard 10-20 EEG system)
    montage = mne.channels.make_standard_montage('standard_1020')
    positions = montage.get_positions()['ch_pos']
    
    # Convert positions from 3D to 2D
    pos_2d = {ch: positions[ch][:2] for ch in electrodes if ch in positions}
    pos_array = np.array(list(pos_2d.values()))
    
    if pos_array.size == 0:
        raise ValueError("No valid electrode positions found for the given data.")
    
    # Create a figure with enough spacing between subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, band in enumerate(frequency_bands):
        if band not in band_averages or not band_averages[band]:
            continue
        
        # Prepare data for the current band
        data = np.array([band_averages[band].get(elec, np.nan) for elec in electrodes if elec in pos_2d])
        if data.size == 0:
            continue
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
    if 'im' in locals():
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Activity Strength', fontsize=14)
    
    plt.show()
    
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



def visualize_long_range_correlations(corr_df):
    """
    Visualizes the strongest correlations between distant EEG electrodes using a network graph.

    Args:
        corr_df (pd.DataFrame): Output DataFrame from `find_strong_long_range_correlations` function.
    """
    # Create a network graph object
    G = nx.Graph()

    # Add nodes and edges with correlation strength as edge weight
    for _, row in corr_df.iterrows():
        elec1, elec2, corr_value = row["Feature 1"].split(".")[1], row["Feature 2"].split(".")[1], row["Correlation"]
        G.add_edge(elec1, elec2, weight=corr_value)

    # Generate node positions using a spring layout for better visualization
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  

    # Draw the network graph
    nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=1200, font_size=10)
    
    # Draw edges with varying width based on correlation strength
    edge_weights = [G[u][v]["weight"] * 5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color="red")

    plt.title("Strongest Long-Range EEG Electrode Correlations")
    plt.show()



def visualize_correlation_gradient(df, threshold=0.5):
    """
    Visualizes the correlation network of all EEG electrodes using a brain-mapped layout,
    with gradient colors representing correlation strength.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    
    # Extract only EEG columns (exclude non-numeric and non-EEG data)
    eeg_columns = [col for col in df.columns if "." in col and df[col].dtype in ["float64", "int64"]]
    if not eeg_columns:
        raise ValueError("No valid EEG columns found in the DataFrame.")
    
    eeg_df = df[eeg_columns].copy()
    
    # Ensure no columns are completely NaN
    eeg_df.dropna(axis=1, how='all', inplace=True)
    if eeg_df.empty:
        raise ValueError("All EEG columns contain only NaN values.")
    
    # Compute correlations for EEG features only
    eeg_corr = eeg_df.corr(method="spearman")
    
    # Filter out weak correlations (keep only |correlation| > threshold)
    strong_corr = eeg_corr.abs().unstack()
    strong_corr = strong_corr[(strong_corr >= threshold) & (strong_corr < 1)]  # Remove self-correlations
    
    if strong_corr.empty:
        raise ValueError("No strong correlations found above the given threshold.")
    
    # Define approximate brain-mapped positions for EEG electrodes
    electrode_positions = {
        "FP1": (-1, 2), "FP2": (1, 2),
        "F3": (-1, 1.5), "Fz": (0, 1.5), "F4": (1, 1.5),
        "C3": (-1, 1), "Cz": (0, 1), "C4": (1, 1),
        "P3": (-1, 0.5), "Pz": (0, 0.5), "P4": (1, 0.5),
        "O1": (-1, 0), "Oz": (0, 0), "O2": (1, 0),
        "T3": (-2, 1), "T4": (2, 1), "T5": (-2, 0.5), "T6": (2, 0.5)
    }
    
    # Create a graph object
    G = nx.Graph()
    edge_weights = []
    edge_colors = []
    
    for (feature1, feature2), corr_value in strong_corr.items():
        try:
            elec1, elec2 = feature1.split(".")[1], feature2.split(".")[1]  # Extract electrode names
            if elec1 in electrode_positions and elec2 in electrode_positions:
                G.add_edge(elec1, elec2, weight=corr_value)
                edge_weights.append(corr_value * 5)  # Scale thickness
                edge_colors.append(corr_value)  # Store correlation for color mapping
        except IndexError:
            continue
    
    if G.number_of_edges() == 0:
        raise ValueError("No valid electrode correlations to visualize.")
    
    # Define node positions based on approximate brain locations
    pos = {node: electrode_positions[node] for node in G.nodes()}
    
    # Create colormap for edge colors
    cmap = plt.cm.plasma
    norm = mcolors.Normalize(vmin=threshold, vmax=1)
    
    # Create figure and axis for colorbar
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw nodes
    nx.draw(G, pos, ax=ax, with_labels=True, node_color="lightblue", node_size=1200, font_size=10)
    
    # Convert edge colors into RGBA values using the colormap
    edge_colors_rgba = [cmap(norm(c)) for c in edge_colors]
    
    # Draw edges with varying thickness and color
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors_rgba, width=edge_weights)
    
    # Create a separate axis for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("EEG Electrode Correlation Strength (Spearman)", fontsize=12)
    
    plt.title(f"Brain-Mapped EEG Electrode Correlation Network (|ρ| > {threshold:.2f})")
    plt.show()
    

def visualize_correlation_gradient_by_disorders(df, disorder1, disorder2, threshold=0.5):
    """
    Visualizes the correlation network of EEG electrodes for two specified main.disorders,
    displaying them side by side.

    Args:
        df (pd.DataFrame): Cleaned EEG dataset.
        disorder1 (str): The first main disorder to filter by.
        disorder2 (str): The second main disorder to filter by.
        threshold (float): Minimum absolute correlation value to visualize an edge.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    if 'main.disorder' not in df.columns:
        raise ValueError("The DataFrame must contain a 'main.disorder' column.")
    
    eeg_columns = [col for col in df.columns if '.' in col and df[col].dtype in ['float64', 'int64']]
    if not eeg_columns:
        raise ValueError("No valid EEG columns found in the DataFrame.")
    
    if df[eeg_columns].isna().all().all():
        raise ValueError("All EEG columns contain only NaN values.")
    
    def compute_graph(df, disorder):
        disorder_df = df[df['main.disorder'] == disorder]
        if disorder_df.empty:
            return None, None, None, None
        
        eeg_df = disorder_df[eeg_columns]
        eeg_corr = eeg_df.corr(method="spearman")
        
        strong_corr = eeg_corr.abs().unstack()
        strong_corr = strong_corr[(strong_corr >= threshold) & (strong_corr < 1)]
        
        if strong_corr.empty:
            return None, None, None, None
        
        G = nx.Graph()
        edge_weights = []
        edge_colors = []
        
        electrode_positions = {
            "FP1": (-1, 2), "FP2": (1, 2),
            "F3": (-1, 1.5), "Fz": (0, 1.5), "F4": (1, 1.5),
            "C3": (-1, 1), "Cz": (0, 1), "C4": (1, 1),
            "P3": (-1, 0.5), "Pz": (0, 0.5), "P4": (1, 0.5),
            "O1": (-1, 0), "Oz": (0, 0), "O2": (1, 0),
            "T3": (-2, 1), "T4": (2, 1), "T5": (-2, 0.5), "T6": (2, 0.5)
        }

        for (feature1, feature2), corr_value in strong_corr.items():
            try:
                elec1, elec2 = feature1.split(".")[1], feature2.split(".")[1]
                if elec1 in electrode_positions and elec2 in electrode_positions:
                    G.add_edge(elec1, elec2, weight=corr_value)
                    edge_weights.append(corr_value * 5)
                    edge_colors.append(corr_value)
            except IndexError:
                continue
        
        pos = {node: electrode_positions[node] for node in G.nodes()}
        return G, pos, edge_weights, edge_colors
    
    G1, pos1, edge_weights1, edge_colors1 = compute_graph(df, disorder1)
    G2, pos2, edge_weights2, edge_colors2 = compute_graph(df, disorder2)
    
    if G1 is None or G2 is None:
        raise ValueError("One or both disorders have no valid correlation data to visualize.")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    cmap = plt.cm.plasma
    norm = mcolors.Normalize(vmin=threshold, vmax=1)

    for ax, G, pos, edge_weights, edge_colors, title in zip(
        axes, [G1, G2], [pos1, pos2], [edge_weights1, edge_weights2], [edge_colors1, edge_colors2],
        [disorder1, disorder2]
    ):
        nx.draw(G, pos, ax=ax, with_labels=True, node_color="lightblue", node_size=1200, font_size=10)
        edge_colors_rgba = [cmap(norm(c)) for c in edge_colors]
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors_rgba, width=edge_weights)
        ax.set_title(f"EEG Correlation Map: {title}")
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("EEG Electrode Correlation Strength (Spearman)", fontsize=12)
    plt.show()


def plot_brain_activity_by_disorder(df, main_disorder):
    """
    Plots average brain activity for all frequency bands for specific disorders within a main disorder.

    Args:
        df (pd.DataFrame): The dataset.
        main_disorder (str): The main disorder to filter by.

    Returns:
        None: Displays a single figure with subplots for all frequency bands.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    if 'main.disorder' not in df.columns or 'specific.disorder' not in df.columns:
        raise ValueError("The DataFrame must contain 'main.disorder' and 'specific.disorder' columns.")
    
    valid_bands = ["delta", "theta", "alpha", "beta", "gamma", "highbeta"]
    band_columns = [col for col in df.columns if any(col.startswith(band) for band in valid_bands)]
    
    if not band_columns:
        raise ValueError("No valid EEG frequency band columns found in the DataFrame.")
    
    if df[band_columns].isna().all().all():
        raise ValueError("All EEG frequency band columns contain only NaN values.")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.flatten()
    
    filtered_df = df[df['main.disorder'] == main_disorder]
    if filtered_df.empty:
        raise ValueError(f"No data found for main disorder: {main_disorder}")
    
    for i, frequency_band in enumerate(valid_bands):
        ax = axes[i]
        band_columns = [col for col in df.columns if col.startswith(frequency_band)]
        
        if not band_columns:
            ax.set_visible(False)
            continue
        
        mean_activity = (
            filtered_df.groupby('specific.disorder')[band_columns]
            .mean()
            .mean(axis=1)
            .sort_values(ascending=False)
        )
        
        if mean_activity.empty:
            ax.set_visible(False)
            continue
        
        mean_activity.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
        ax.set_title(f"{frequency_band.capitalize()} Activity")
        ax.set_xlabel("Specific Disorder")
        ax.set_ylabel("Average Activity")
        ax.tick_params(axis='x', rotation=45)
    
    for j in range(len(valid_bands), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f"Brain Activity by Frequency Band (Main Disorder: {main_disorder})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
