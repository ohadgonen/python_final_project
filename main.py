# here we will call all of the functions.

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

eeg_file = '/Users/ohadgonen/Desktop/Neuroscience/Year 2/1st semester/Advenced programming in Python/מטלות בית/python_final_project/src/EEG.machinelearing_data_BRMH.csv'  
df = pd.read_csv(eeg_file)

# DATA CLEANING AND INITIAL EXPLORATION
# Insert tests!!! for example:
# from tests.test_data_cleaning import test_fill_NaNs as test_fill_NaNs
# test_fill_NaNs(cleaned_df)


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
cleaned_df = cleaned_df.drop(columns=['Unnamed: 122'])

# Next, we'll remove any duplicate rows.
cleaned_df = cleaned_df.drop_duplicates()

# Standarize all of the categorical columns in df (by converting all text to lowercase and stripping any leading/trailing spaces).
from src.data_cleaning import standardize_categorical_columns as standardize_categorical_columns
cleaned_df = standardize_categorical_columns(cleaned_df)

# Check for outliers in the categorical columns. 
from src.data_cleaning import check_for_categorical_outliers as check_for_categorical_outliers
check_for_categorical_outliers(cleaned_df)

# There aren't any categorical outliers in the categorical columns.
# We will leave all of the values in the eeg data columns because we don't wish to modify it. 

# Update the original dataframe, after the cleaning.
df = cleaned_df

# DATA ANALYSIS

# What are the most frequently coupled diagnoses?
from src.data_analysis import find_most_frequent_coupled_diagnoses as find_most_frequent_coupled_diagnoses
# Get the most common pairs as a DataFrame
most_common_pairs_df = find_most_frequent_coupled_diagnoses(df, 'main.disorder', 'specific.disorder')
print("\n", most_common_pairs_df)

from src.data_visualization import visualize_correlation as visualize_correlation
# Visualize correlation for selected columns
visualize_correlation(df, cols=['education', 'IQ'])
























