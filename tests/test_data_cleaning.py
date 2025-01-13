import sys
import os
import numpy as np
import pandas as pd

from src.data_cleaning import fill_NaNs as fill_NaNs

def test_fill_NaNs():

    # Positive Test Case
    print("Running Positive Test Case...")
    df_positive = pd.DataFrame({
        'col1': [1, 2, 3, None, 5],
        'col2': [None, 2, None, 4, None]
    })
    result_positive = fill_NaNs(df_positive, 'col1')
    assert pd.isna(result_positive['col1']).sum() == 0, "Positive Test Case Failed: NaNs not filled correctly"
    assert result_positive['col1'].iloc[3] == df_positive['col1'].mean(), \
        "Positive Test Case Failed: NaNs not filled with the correct value"
    print("Positive Test Case Passed!")

    # Negative Test Case
    print("Running Negative Test Case...")
    df_negative = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    try:
        result_negative = fill_NaNs(df_negative, 'col3')  # Non-existent column
        assert 'col3' not in result_negative.columns, \
            "Negative Test Case Failed: Non-existent column handled incorrectly"
        print("Negative Test Case Passed!")
    except Exception as e:
        print(f"Negative Test Case Failed: {e}")

    # Boundary Test Case
    print("Running Boundary Test Case...")
    df_boundary = pd.DataFrame({
        'col1': [float('inf'), 1, None, 5, float('-inf')],
    })
    result_boundary = fill_NaNs(df_boundary, 'col1')
    assert pd.isna(result_boundary['col1']).sum() == 0, "Boundary Test Case Failed: NaNs not filled correctly"
    assert result_boundary['col1'].iloc[2] == df_boundary['col1'].mean(), \
        "Boundary Test Case Failed: NaNs not filled with the correct value"
    print("Boundary Test Case Passed!")

    # Error Test Case
    print("Running Error Test Case...")
    df_error = pd.DataFrame({'col1': [1, 2, 3]})
    try:
        result_error = fill_NaNs(None, 'col1')  # Invalid input: None instead of DataFrame
        print("Error Test Case Failed: Invalid input handled incorrectly")
    except AttributeError:
        print("Error Test Case Passed!")

    # Null Test Case
    print("Running Null Test Case...")
    df_null = pd.DataFrame({'col1': []})  # Empty DataFrame
    result_null = fill_NaNs(df_null, 'col1')
    assert result_null.empty, "Null Test Case Failed: Empty DataFrame handled incorrectly"
    print("Null Test Case Passed!")

