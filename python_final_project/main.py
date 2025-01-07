# here we will call all of the functions.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

eeg_file = '/Users/ohadgonen/Desktop/Neuroscience/Year 2/1st semester/Advenced programming in Python/מטלות בית/python_final_project/src/EEG.machinelearing_data_BRMH.csv'  
df = pd.read_csv(eeg_file)

from src.data_analysis import evaluate_lobe_activity as ela
# Example usage
print(ela(df, 12, 'delta', 'occipital'))






