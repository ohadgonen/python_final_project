import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import mne 
warnings.simplefilter(action='ignore', category=FutureWarning)

eeg_file = '/Users/ohadgonen/Desktop/Neuroscience/Year 2/1st semester/Advenced programming in Python/מטלות בית/python_final_project/src/EEG.machinelearing_data_BRMH.csv'  
df = pd.read_csv(eeg_file)


# What kind of information are we dealing with? let's get some intuition about the participants.

from src.data_visualization import visualize_non_eeg_data as visualize_non_eeg_data
visualize_non_eeg_data(df)

# We can see there are some missing values in the education and the IQ columns.
# Since there are 15 and 13 NaNs, numbers which are negligible to the 945 rows, we will fill each of the missing values with the average of the column.
from src.data_cleaning import fill_NaNs as fill_NaNs
cleaned_df = fill_NaNs(df,'education')
cleaned_df = fill_NaNs(df, 'IQ')

# Now let's look at the eeg part of the dataframe.
from src.data_visualization import visualize_eeg_data as visualize_eeg_data
visualize_eeg_data(df)

# Check that the df includes no NaNs.
from src.data_cleaning import check_missing_electrode_values as check_missing_electrode_values
check_missing_electrode_values(df)

# We've found that column 122 is full of NaNs, hence it doesn't contain any useful data. Let's delete the column.
cleaned_df = cleaned_df.dropna(axis=1, how='all')

# Run check_missing_electrode_values to insure there are no missing values in the dataframe.
check_missing_electrode_values(df)

# Drop all columns that start with 'COH' since they are not relevant to our analysis. We want to work with the raw EEG data. 
cleaned_df = cleaned_df.loc[:, ~cleaned_df.columns.str.startswith('COH')]

# Next, we'll remove any duplicate rows.
cleaned_df = cleaned_df.drop_duplicates()

# Standarize all of the categorical columns in df (by converting all text to lowercase and stripping any leading/trailing spaces).
from src.data_cleaning import standardize_categorical_columns as standardize_categorical_columns
cleaned_df = standardize_categorical_columns(cleaned_df)

# Check for outliers in the categorical columns. 
from src.data_cleaning import check_for_categorical_outliers as check_for_categorical_outliers
check_for_categorical_outliers(cleaned_df)

# There aren't any categorical outliers in the categorical columns.

# Reformat the electrode columns from prefix.band.type.channel to band.channel.
from src.data_cleaning import reformat_electrode_columns as reformat_electrode_columns
cleaned_df = reformat_electrode_columns(cleaned_df)
cleaned_df.head()

# We will leave all of the values in the eeg data columns because we don't wish to modify it.

# Update the original dataframe, after the cleaning.
df = cleaned_df


''' Q1: What is the characteristic EEG activity of different main psychiatric disorders? '''

# Let's start by checking the distribution of the main psychiatric disorder column.
from src.data_visualization import visualize_main_psychiatric_disorders as visualize_main_psychiatric_disorders
print(df['main.disorder'].unique())
visualize_main_psychiatric_disorders(df)

from src.data_analysis import calculate_band_averages as calculate_band_averages
# Choose an example main disorder.
main_disorder = "schizophrenia"
# Create a dictionary of average electrode activity in each band for schizophrenia. 
band_averages = calculate_band_averages(main_disorder, df)
# Display the result dictionary's keys
band_averages.keys()

from src.data_visualization import visualize_brain_activity as visualize_brain_activity
# Visualize the brain activity for the given band averages in schizophrenia.
visualize_brain_activity(band_averages, main_disorder)

# We will try to look for significant differences in the brain activity of main.disorder vs healthy control. 
# We will take schizophrenia as a first example. 
from src.data_analysis import find_significant_differences as find_significant_differences

hc = "healthy control"
main_disorder = "schizophrenia"
# Create a dictionary of average electrode activity in each band for healthy control. 
healthy_band_averages = calculate_band_averages(hc, df)
# Create a dictionary of average electrode activity in each band for schizophrenia.
disorder_band_averages = calculate_band_averages(main_disorder, df)

significant_differences = find_significant_differences(df, main_disorder, hc,  p_threshold=0.0001, activity_type= "enhanced")
print(significant_differences)

# Let's try to examine another disorder.
main_disorder = "mood disorder"
# Create a dictionary of average electrode activity in each band for mood disorder. 
band_averages = calculate_band_averages(main_disorder, df)
# Visualize the brain activity for the given band averages in mood disorder.
visualize_brain_activity(band_averages, main_disorder)

# Find significant enhanced activity in mood disorder compared to healthy control. 
disorder_band_averages = calculate_band_averages(main_disorder, df)
significant_differences = find_significant_differences(df, main_disorder, hc,  p_threshold=0.01, activity_type= "enhanced")
print(significant_differences)

# Let's search for areas and frequency bands with reduced activity in mood disorder.
significant_differences = find_significant_differences(df, main_disorder, hc,  p_threshold=0.05, activity_type ="lower")
print(significant_differences)

# Try addictive disorder as an example for reduced activity. 
main_disorder = "addictive disorder"
# p = 0.28 is the lowest value we have found that gives us an electrode with reduced activity. 
significant_differences = find_significant_differences(df, main_disorder, hc,  p_threshold=0.28, activity_type ="lower")
print(significant_differences)

# Visualize the EEG head maps for all disorders. 

from src.data_analysis import prepare_disorder_band_averages as prepare_disorder_band_averages
# Prepare the required arguments for the `visualize_all_disorders` function from the given dataset.
disorders = df['main.disorder'].unique().tolist()
frequency_bands = ["delta", "theta", "alpha", "beta", "gamma", "highbeta"]
electrodes = ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 
              'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']

# Generate the required arguments
disorder_band_averages, disorder_names = prepare_disorder_band_averages(df, disorders, frequency_bands, electrodes)

# Visualize the brain activity for all main disorders in the dataset.
from src.data_visualization import visualize_all_disorders as visualize_all_disorders

visualize_all_disorders(disorder_band_averages, disorder_names)

# The strength of of the EEG activity is normalized compared to the minimun and maximun values of the band averages.


'''Q2: How do different specific disorders vary in brain activity across various frequency bands within the same main disorder?''' 

from src.data_visualization import plot_brain_activity_by_disorder
main_disorder = "trauma and stress related disorder"
plot_brain_activity_by_disorder(df,main_disorder)

main_disorder = "addictive disorder"
plot_brain_activity_by_disorder(df,main_disorder)


'''Q3: Are there any strong correlations between distant (not adjacent) electrodes in healthy controls? '''

# We defined distant electrodes as pairs of electrodes located in different brain regions that are anatomically and functionally separated.
# Specifically, we consider electrodes distant if they belong to one of the following region pairs:
# Frontal–Occipital, Frontal–Temporal, Central–Occipital, Temporal–Parietal, or Temporal–Occipital. 
# This ensures that only true long-range correlations are analyzed, excluding nearby or highly connected regions.

# What are the strongest correlations between distant electrodes in healthy controls?
from src.data_analysis import find_strong_long_range_correlations as find_strong_long_range_correlations

# Create a df that contains only healthy controls. 
disorder = "healthy control"
hc_df = df[df['main.disorder'] == disorder]

# Run the function to find the top 45 strongest long-range correlations.
top_long_range_corr_df = find_strong_long_range_correlations(hc_df, num_pairs=45, threshold=0.6)

# Display the resulting DataFrame.
top_long_range_corr_df.head()

# Visualize the long-range correlations. 
from src.data_visualization import visualize_long_range_correlations as visualize_long_range_correlations
visualize_long_range_correlations(top_long_range_corr_df)

# Let's try to map all of the strong correlations (>0.5) between all electrodes of healthy controls patients. 
from src.data_visualization import visualize_correlation_gradient as visualize_correlation_gradient
visualize_correlation_gradient(hc_df)

# How do correlation patterns differ between healthy controls and patients with OCD?
# We can use our correlation measure to compare these two groups. 
from src.data_visualization import visualize_correlation_gradient_by_disorders as visualize_correlation_gradient_by_disorders
visualize_correlation_gradient_by_disorders(df,'healthy control', 'obsessive compulsive disorder')


'''Q4: Can we predict the main psychiatric disorder based on the EEG data?'''

from src.model import split_data, train_model, evaluate_model, preprocess_data

# Running the pipeline
df_processed, label_encoders = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(df_processed)
rf_model = train_model(X_train, y_train)
accuracy, classification_rep = evaluate_model(rf_model, X_test, y_test, label_encoders["main.disorder"])

# Convert classification report dictionary to a DataFrame
classification_df = pd.DataFrame.from_dict(classification_rep).transpose()

# Print accuracy and classification report as a formatted table
print(f"\nModel Accuracy: {accuracy:.2f}\n")
print("Classification Report:")
print(classification_df.to_string()) 

# We have achieved an accuracy of 0.34, which is not very high. 