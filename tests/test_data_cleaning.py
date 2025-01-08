import numpy as np
import pandas as pd

from src.data_cleaning import fill_NaNs as fill_NaNs

def test_fill_NaNs(df):
    # Display the original DataFrame with NaNs before filling
    print("Original DataFrame with NaNs:")
    print(df)
    
    # Fill NaNs in 'education' and 'IQ' columns
    df = fill_NaNs(df, 'education')
    df = fill_NaNs(df, 'IQ')
    
    # Display the DataFrame after filling NaNs
    print("\nDataFrame after filling NaNs:")
    print(df)
    
    # Verify that there are no NaNs in the DataFrame after applying the fill_NaNs function
    assert df.isna().sum().sum() == 0, "There are still NaNs in the DataFrame after filling"
    
    print("\nTest passed: No NaNs remain in the DataFrame.")

