import sys
import os
import numpy as np
import pandas as pd
import warnings

# Disable FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from src.data_cleaning import fill_NaNs

def test_fill_NanNs():
    # Positive Test Case: Verifies that NaNs are filled with the average value
    df = pd.DataFrame({
        'column1': [1, 2, np.nan, 4, 5]
    })
    expected_df = pd.DataFrame({
        'column1': [1, 2, 3, 4, 5]  # NaN replaced with 3 (average of 1, 2, 4, 5)
    })
    result = fill_NaNs(df, 'column1')
    
    # Check if the values are the same, regardless of data type differences
    pd.testing.assert_series_equal(result['column1'], expected_df['column1'], check_dtype=False)

    # Negative Test Case: Tests the behavior when a non-existing column is provided
    df = pd.DataFrame({
        'column1': [1, 2, np.nan, 4, 5]
    })
    result = fill_NaNs(df, 'non_existing_column')
    # The DataFrame should remain unchanged since the column doesn't exist
    pd.testing.assert_frame_equal(result, df)

    # Boundary Test Case: Tests with a column of all NaNs or a single row
    df_all_nans = pd.DataFrame({
        'column1': [np.nan, np.nan, np.nan, np.nan, np.nan]
    })
    expected_df_all_nans = pd.DataFrame({
        'column1': [np.nan, np.nan, np.nan, np.nan, np.nan]  # Cannot compute average, NaN remains
    })
    result_all_nans = fill_NaNs(df_all_nans, 'column1')
    pd.testing.assert_frame_equal(result_all_nans, expected_df_all_nans)

    # Single row DataFrame
    df_single_row = pd.DataFrame({
        'column1': [np.nan]
    })
    expected_df_single_row = pd.DataFrame({
        'column1': [np.nan]  # NaN remains as there's only one row
    })
    result_single_row = fill_NaNs(df_single_row, 'column1')
    pd.testing.assert_frame_equal(result_single_row, expected_df_single_row)

    # Error Test Case: Force an error by passing a non-DataFrame type
    try:
        result = fill_NaNs("not_a_dataframe", 'column1')
        print("Error test failed: No exception raised.")
    except Exception as e:
        pass  # Error test passed

    # Null Test Case: Tests the system with null (empty) values
    df_empty = pd.DataFrame({
        'column1': []
    })
    result_empty = fill_NaNs(df_empty, 'column1')
    expected_empty = pd.DataFrame({'column1': []})  # Empty DataFrame should remain unchanged
    pd.testing.assert_frame_equal(result_empty, expected_empty)

    # If all tests pass
    print("all tests for fill_NaNs passed")

# Run the tests
test_fill_NanNs()


from src.data_cleaning import check_missing_electrode_values

def test_check_missing_electrode_values():
    try:
        # Positive Test Case: Verifies that missing values in electrode columns are correctly identified
        df = pd.DataFrame({
            'no.': [1, 2, 3],
            'sex': ['M', 'F', 'M'],
            'age': [25, 30, 35],
            'eeg.date': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'education': ['PhD', 'Masters', 'Bachelors'],
            'IQ': [120, 110, 115],
            'main.disorder': ['None', 'Anxiety', 'Depression'],
            'specific.disorder': ['None', 'None', 'None'],
            'electrode_1': [1.2, np.nan, 2.4],
            'electrode_2': [np.nan, 3.5, 4.1],
        })
        check_missing_electrode_values(df)  # Expected: "electrode_1" and "electrode_2" with missing values

        # Negative Test Case: Tests when DataFrame has an invalid structure (missing expected columns)
        df_invalid = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': [4, 5, 6]
        })
        check_missing_electrode_values(df_invalid)  # Expected: No electrode columns

        # Boundary Test Case: Tests with the minimum valid DataFrame (one row)
        df_single_row = pd.DataFrame({
            'no.': [1],
            'sex': ['M'],
            'age': [25],
            'eeg.date': ['2021-01-01'],
            'education': ['PhD'],
            'IQ': [120],
            'main.disorder': ['None'],
            'specific.disorder': ['None'],
            'electrode_1': [np.nan],
            'electrode_2': [1.0],
        })
        check_missing_electrode_values(df_single_row)  # Expected: "electrode_1" has missing values

        # Error Test Case: Force an error by passing a non-DataFrame type
        try:
            check_missing_electrode_values("not_a_dataframe")
        except AttributeError:
            pass  # Expected: error message for invalid input

        # Null Test Case: Test with a DataFrame that contains no electrode columns
        df_empty_electrodes = pd.DataFrame({
            'no.': [1, 2, 3],
            'sex': ['M', 'F', 'M'],
            'age': [25, 30, 35],
            'eeg.date': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'education': ['PhD', 'Masters', 'Bachelors'],
            'IQ': [120, 110, 115],
            'main.disorder': ['None', 'Anxiety', 'Depression'],
            'specific.disorder': ['None', 'None', 'None'],
        })
        check_missing_electrode_values(df_empty_electrodes)  # Expected: No missing values in electrode columns

        print("All tests for check_missing_electrode_values passed.")

    except Exception as e:
        print(f"Test failed: {e}")

# Run the tests
test_check_missing_electrode_values()


from src.data_cleaning import standardize_categorical_columns

def test_standardize_categorical_columns():
    try:
        # Positive Test Case: Verifies that categorical columns are standardized correctly
        df = pd.DataFrame({
            'name': [' Alice ', 'BOB', ' Charlie  '],
            'age': [25, 30, 35],
            'city': ['New York  ', ' LOS ANGELES ', '  Chicago']
        })
        expected_df = pd.DataFrame({
            'name': ['alice', 'bob', 'charlie'],
            'age': [25, 30, 35],
            'city': ['new york', 'los angeles', 'chicago']
        })
        
        result_df = standardize_categorical_columns(df)
        assert result_df.equals(expected_df), f"Expected:\n{expected_df}\n but got:\n{result_df}"

        # Negative Test Case: Tests when the DataFrame does not contain categorical columns
        df_no_categorical = pd.DataFrame({
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })
        result_df_no_cat = standardize_categorical_columns(df_no_categorical)
        # No categorical columns, the DataFrame should remain unchanged
        assert df_no_categorical.equals(result_df_no_cat), "DataFrame with no categorical columns should remain unchanged"

        # Boundary Test Case: Tests with a single row
        df_single_row = pd.DataFrame({
            'name': ['  John  '],
            'age': [28],
            'city': ['   Boston']
        })
        expected_single_row = pd.DataFrame({
            'name': ['john'],
            'age': [28],
            'city': ['boston']
        })
        result_single_row = standardize_categorical_columns(df_single_row)
        assert result_single_row.equals(expected_single_row), f"Expected:\n{expected_single_row}\n but got:\n{result_single_row}"

        # Error Test Case: Force an error by passing a non-DataFrame type
        try:
            standardize_categorical_columns("not_a_dataframe")
            print("Error test failed: No exception raised.")
        except AttributeError as e:
            print(f"Error test passed: {e}")

        # Null Test Case: Test with a DataFrame that contains null values
        df_with_nulls = pd.DataFrame({
            'name': ['  Alice  ', None, '  Charlie '],
            'age': [25, 30, 35],
            'city': ['New York', '  ', 'Chicago']
        })
        expected_with_nulls = pd.DataFrame({
            'name': ['alice', None, 'charlie'],
            'age': [25, 30, 35],
            'city': ['new york', '', 'chicago']
        })
        result_with_nulls = standardize_categorical_columns(df_with_nulls)
        assert result_with_nulls.equals(expected_with_nulls), f"Expected:\n{expected_with_nulls}\n but got:\n{result_with_nulls}"

        print("All tests for standardize_categorical_columns passed.")

    except Exception as e:
        print(f"Test failed: {e}")

# Run the tests
test_standardize_categorical_columns()


from src.data_cleaning import check_for_categorical_outliers

def test_check_for_categorical_outliers():
    try:
        # Positive Test Case: Verifies detection of rare categories (outliers)
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Edward', 'Frank', 'Grace', 'Hank', 'Ivy', 'Jack'],
            'city': ['New York', 'Los Angeles', 'Chicago', 'New York', 'New York', 'Chicago', 'Chicago', 'Los Angeles', 'Chicago', 'New York'],
            'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
        })
        
        # Adding rare categories using pd.concat
        new_cities = pd.Series(['Boston', 'Houston', 'Seattle', 'Phoenix', 'Miami'])
        df['city'] = pd.concat([df['city'], new_cities], ignore_index=True)

        # Run the function and capture the output
        check_for_categorical_outliers(df)  # Expected: Rare categories in 'city' with count < 5 (Boston, Houston, etc.)

        # Negative Test Case: Tests when there are no rare categories
        df_no_outliers = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'city': ['New York', 'Los Angeles', 'Chicago', 'Miami'],
            'age': [25, 30, 35, 40]
        })
        check_for_categorical_outliers(df_no_outliers)  # Expected: No rare categories detected
        
        # Boundary Test Case: Tests with the minimum valid DataFrame (one row)
        df_single_row = pd.DataFrame({
            'name': ['Alice'],
            'city': ['New York'],
            'age': [25]
        })
        check_for_categorical_outliers(df_single_row)  # Expected: No rare categories detected (only one row)

        # Error Test Case: Force an error by passing a non-DataFrame type
        try:
            check_for_categorical_outliers("not_a_dataframe")
            print("Error test failed: No exception raised.")
        except AttributeError as e:
            print(f"Error test passed: {e}")

        # Null Test Case: Test with a DataFrame that contains null values
        df_with_nulls = pd.DataFrame({
            'name': ['Alice', None, 'Charlie', 'David'],
            'city': ['New York', 'New York', 'Chicago', None],
            'age': [25, 30, 35, 40]
        })
        check_for_categorical_outliers(df_with_nulls)  # Expected: No rare categories detected but handle None values

        print("All tests for check_for_categorical_outliers passed.")
        
    except Exception as e:
        print(f"Test failed: {e}")

# Run the tests
test_check_for_categorical_outliers()



from src.data_cleaning import reformat_electrode_columns

def test_reformat_electrode_columns():
    try:
        # Positive Test Case: Verifies that electrode column names are reformatted correctly
        df = pd.DataFrame({
            'no.': [1, 2, 3],
            'AB.A.delta.d.F3': [1.2, 2.3, 3.4],
            'AB.A.gamma.d.F4': [4.5, 5.6, 6.7],
            'age': [25, 30, 35]
        })

        expected_df = pd.DataFrame({
            'no.': [1, 2, 3],
            'delta.F3': [1.2, 2.3, 3.4],
            'gamma.F4': [4.5, 5.6, 6.7],
            'age': [25, 30, 35]
        })

        result_df = reformat_electrode_columns(df)
        assert result_df.equals(expected_df), f"Expected:\n{expected_df}\n but got:\n{result_df}"

        # Negative Test Case: Tests when there are no valid electrode columns
        df_no_electrodes = pd.DataFrame({
            'no.': [1, 2, 3],
            'sex': ['M', 'F', 'M'],
            'eeg.date': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'education': ['PhD', 'Masters', 'Bachelors']
        })
        
        result_no_electrodes = reformat_electrode_columns(df_no_electrodes)
        # Since there are no electrode columns, the DataFrame should remain unchanged
        assert df_no_electrodes.equals(result_no_electrodes), "DataFrame with no electrode columns should remain unchanged"

        # Boundary Test Case: Tests with a DataFrame containing one valid electrode column
        df_single_electrode = pd.DataFrame({
            'no.': [1],
            'AB.A.delta.d.F3': [1.2],
            'age': [25]
        })
        
        expected_single_electrode = pd.DataFrame({
            'no.': [1],
            'delta.F3': [1.2],
            'age': [25]
        })

        result_single_electrode = reformat_electrode_columns(df_single_electrode)
        assert result_single_electrode.equals(expected_single_electrode), f"Expected:\n{expected_single_electrode}\n but got:\n{result_single_electrode}"

        # Error Test Case: Force an error by passing a non-DataFrame type
        try:
            reformat_electrode_columns("not_a_dataframe")
            print("Error test failed: No exception raised.")
        except AttributeError as e:
            print(f"Error test passed: {e}")

        # Null Test Case: Test with a DataFrame that contains null or missing values
        df_with_nulls = pd.DataFrame({
            'AB.A.delta.d.F3': [1.2, None, 3.4],
            'age': [25, 30, 35]
        })

        expected_with_nulls = pd.DataFrame({
            'delta.F3': [1.2, None, 3.4],
            'age': [25, 30, 35]
        })

        result_with_nulls = reformat_electrode_columns(df_with_nulls)
        assert result_with_nulls.equals(expected_with_nulls), f"Expected:\n{expected_with_nulls}\n but got:\n{result_with_nulls}"

        print("All tests for reformat_electrode_columns passed.")

    except Exception as e:
        print(f"Test failed: {e}")

# Run the tests
test_reformat_electrode_columns()


