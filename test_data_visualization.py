import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import mne

from src.data_visualization import (
visualize_non_eeg_data,
visualize_eeg_data,
visualize_main_psychiatric_disorders,
visualize_brain_activity,
visualize_all_disorders,
visualize_long_range_correlations,
visualize_correlation_gradient,
visualize_correlation_gradient_by_disorders,
plot_brain_activity_by_disorder
)



def test_visualize_non_eeg_data():
    """
    Test cases for visualize_non_eeg_data to ensure handling of positive, negative, null, error, and boundary conditions.
    """
    all_tests_passed = True
    
    # Positive Test: Properly formatted DataFrame
    df_positive = pd.DataFrame({
        'no.': range(5),
        'sex': ['M', 'F', 'M', 'F', 'M'],
        'age': [25, 30, 35, 40, 45],
        'eeg.date': pd.date_range(start='1/1/2022', periods=5),
        'education': [12, 16, 14, 18, 20],
        'IQ': [110, 120, 130, 140, 100],
        'main.disorder': ['Anxiety', 'Depression', 'Bipolar', 'OCD', 'PTSD'],
        'specific.disorder': ['GAD', 'MDD', 'BP1', 'OCD', 'PTSD']
    })
    try:
        visualize_non_eeg_data(df_positive)
    except Exception as e:
        print("Failed Positive Test:", e)
        all_tests_passed = False
    
    # Negative Test: Missing critical columns
    df_negative = pd.DataFrame({
        'no.': range(5),
        'sex': ['M', 'F', 'M', 'F', 'M'],
        'age': [25, 30, 35, 40, 45]
    })
    try:
        visualize_non_eeg_data(df_negative)
        print("Failed Negative Test: Should have raised an error")
        all_tests_passed = False
    except Exception:
        pass
    
    # Null Test: Some null values in the DataFrame
    df_null = df_positive.copy()
    df_null.loc[2, 'age'] = None
    df_null.loc[3, 'IQ'] = None
    try:
        visualize_non_eeg_data(df_null)
    except Exception as e:
        print("Failed Null Test:", e)
        all_tests_passed = False
    
    # Error Test: Non-DataFrame input
    try:
        visualize_non_eeg_data("Not a DataFrame")
        print("Failed Error Test: Should have raised an error")
        all_tests_passed = False
    except Exception:
        pass
    
    # Boundary Test: Minimum valid dataset (only one row)
    df_boundary = pd.DataFrame({
        'no.': [1],
        'sex': ['M'],
        'age': [22],
        'eeg.date': [pd.Timestamp('2022-01-01')],
        'education': [12],
        'IQ': [100],
        'main.disorder': ['Anxiety'],
        'specific.disorder': ['GAD']
    })
    try:
        visualize_non_eeg_data(df_boundary)
    except Exception as e:
        print("Failed Boundary Test:", e)
        all_tests_passed = False
    
    if all_tests_passed:
        print("visualize_non_eeg_data: Passed all tests")

if __name__ == "__main__":
    test_visualize_non_eeg_data()


def test_visualize_eeg_data():
    """
    Test cases for visualize_eeg_data to ensure handling of positive, negative, null, error, and boundary conditions.
    """
    all_tests_passed = True
    
    # Positive Test: Properly formatted DataFrame with EEG columns
    df_positive = pd.DataFrame(
        np.random.rand(5, 5),
        columns=['Fz', 'Cz', 'Pz', 'O1', 'O2']
    )
    df_positive.insert(0, 'no.', range(5))
    try:
        visualize_eeg_data(df_positive)
    except Exception as e:
        print("Failed Positive Test:", e)
        all_tests_passed = False
    
    # Negative Test: Missing EEG columns
    df_negative = pd.DataFrame({'no.': range(5)})
    try:
        visualize_eeg_data(df_negative)
        print("Failed Negative Test: Should have raised an error")
        all_tests_passed = False
    except Exception:
        pass
    
    # Null Test: Some null values in EEG data
    df_null = df_positive.copy()
    df_null.loc[2, 'Fz'] = None
    df_null.loc[3, 'Cz'] = None
    try:
        visualize_eeg_data(df_null)
    except Exception as e:
        print("Failed Null Test:", e)
        all_tests_passed = False
    
    # Error Test: Non-DataFrame input
    try:
        visualize_eeg_data("Not a DataFrame")
        print("Failed Error Test: Should have raised an error")
        all_tests_passed = False
    except Exception:
        pass
    
    # Boundary Test: Minimum valid dataset (only one row)
    df_boundary = pd.DataFrame(
        np.random.rand(1, 5),
        columns=['Fz', 'Cz', 'Pz', 'O1', 'O2']
    )
    df_boundary.insert(0, 'no.', [1])
    try:
        visualize_eeg_data(df_boundary)
    except Exception as e:
        print("Failed Boundary Test:", e)
        all_tests_passed = False
    
    if all_tests_passed:
        print("visualize_eeg_data: Passed all tests")

if __name__ == "__main__":
    test_visualize_eeg_data()


def test_visualize_main_psychiatric_disorders():
    """
    Test cases for visualize_main_psychiatric_disorders to ensure correct handling of valid and edge cases.
    """
    all_tests_passed = True
    
    # Positive Test: Proper DataFrame with 'main.disorder' column
    df_positive = pd.DataFrame({'main.disorder': np.random.choice(['Anxiety', 'Depression', 'Bipolar'], size=100)})
    try:
        visualize_main_psychiatric_disorders(df_positive)
    except Exception as e:
        print("Failed Positive Test:", e)
        all_tests_passed = False
    
    # Negative Test: Missing 'main.disorder' column
    df_negative = pd.DataFrame({'random_column': np.random.choice(['A', 'B', 'C'], size=100)})
    try:
        visualize_main_psychiatric_disorders(df_negative)
    except Exception:
        pass
    else:
        print("Failed Negative Test: Should have raised an error")
        all_tests_passed = False
    
    if all_tests_passed:
        print("visualize_main_psychiatric_disorders: Passed all tests")

if __name__ == "__main__":
    test_visualize_main_psychiatric_disorders()



def test_visualize_brain_activity():
    """
    Test cases for visualize_brain_activity to ensure handling of positive, negative, null, error, and boundary conditions.
    """
    all_tests_passed = True
    
    # Positive Test: Properly formatted band averages dictionary
    band_averages_positive = {
        'alpha': {'Fz': 0.5, 'Cz': 0.3, 'Pz': 0.7},
        'beta': {'Fz': 0.6, 'Cz': 0.4, 'Pz': 0.8}
    }
    try:
        visualize_brain_activity(band_averages_positive, 'Anxiety')
    except Exception as e:
        print("Failed Positive Test:", e)
        all_tests_passed = False
    
    # Negative Test: Missing frequency bands
    band_averages_negative = {}
    try:
        visualize_brain_activity(band_averages_negative, 'Depression')
    except Exception:
        pass
    else:
        print("Failed Negative Test: Should have raised an error")
        all_tests_passed = False
    
    # Null Test: Only None values should be filtered out correctly
    band_averages_null = {
        'alpha': {'Fz': None, 'Cz': 0.3, 'Pz': None},
        'beta': {'Fz': 0.6, 'Cz': None, 'Pz': 0.8}
    }
    try:
        visualize_brain_activity(band_averages_null, 'PTSD')
    except Exception:
        pass
    else:
        print("Failed Null Test: Should have raised an error")
        all_tests_passed = False
    
    # Boundary Test: Ensure at least one valid electrode exists
    band_averages_boundary = {
        'alpha': {}
    }
    try:
        visualize_brain_activity(band_averages_boundary, 'OCD')
    except Exception:
        pass
    else:
        print("Failed Boundary Test: Should have raised an error")
        all_tests_passed = False
    
    if all_tests_passed:
        print("visualize_brain_activity: Passed all tests")

if __name__ == "__main__":
    test_visualize_brain_activity()



def test_visualize_all_disorders():
    """
    Test cases for visualize_all_disorders to ensure handling of positive, negative, null, error, and boundary conditions.
    """
    all_tests_passed = True
    
    # Positive Test: Properly formatted input with all expected frequency bands
    frequency_bands = ["gamma", "highbeta", "beta", "alpha", "theta", "delta"]
    disorder_band_averages_positive = [
        {band: {'Fz': np.random.rand(), 'Cz': np.random.rand()} for band in frequency_bands} for _ in range(7)
    ]
    disorder_names_positive = ['Anxiety', 'Depression', 'Bipolar', 'OCD', 'PTSD', 'Schizophrenia', 'ADHD']
    try:
        visualize_all_disorders(disorder_band_averages_positive, disorder_names_positive)
    except Exception as e:
        print("Failed Positive Test:", e)
        all_tests_passed = False
    
    # Negative Test: Missing frequency bands
    disorder_band_averages_negative = [{} for _ in range(7)]
    disorder_names_negative = disorder_names_positive
    try:
        visualize_all_disorders(disorder_band_averages_negative, disorder_names_negative)
    except Exception:
        pass
    else:
        print("Failed Negative Test: Should have raised an error")
        all_tests_passed = False
    
    # Null Test: Some None values in disorder bands
    disorder_band_averages_null = [
        {band: {'Fz': None, 'Cz': np.random.rand()} for band in frequency_bands} for _ in range(7)
    ]
    disorder_names_null = disorder_names_positive
    try:
        visualize_all_disorders(disorder_band_averages_null, disorder_names_null)
    except Exception:
        pass
    else:
        print("Failed Null Test: Should have raised an error")
        all_tests_passed = False
    
    # Error Test: Non-list input
    try:
        visualize_all_disorders("Not a list", disorder_names_positive)
    except Exception:
        pass
    else:
        print("Failed Error Test: Should have raised an error")
        all_tests_passed = False
    
    # Boundary Test: Incorrect number of disorders (should be 7)
    disorder_band_averages_boundary = [
        {band: {'Fz': np.random.rand(), 'Cz': np.random.rand()} for band in frequency_bands} for _ in range(6)
    ]
    disorder_names_boundary = disorder_names_positive[:6]
    try:
        visualize_all_disorders(disorder_band_averages_boundary, disorder_names_boundary)
    except Exception:
        pass
    else:
        print("Failed Boundary Test: Should have raised an error")
        all_tests_passed = False
    
    if all_tests_passed:
        print("visualize_all_disorders: Passed all tests")

if __name__ == "__main__":
    test_visualize_all_disorders()



def test_visualize_long_range_correlations():
    """
    Test cases for visualize_long_range_correlations to ensure handling of positive, negative, null, error, and boundary conditions.
    """
    all_tests_passed = True
    
    # Positive Test: Properly formatted correlation DataFrame
    corr_df_positive = pd.DataFrame({
        'Feature 1': ['alpha.Fz', 'beta.Cz', 'theta.Pz'],
        'Feature 2': ['gamma.O1', 'delta.O2', 'beta.F3'],
        'Correlation': [0.8, 0.7, 0.9]
    })
    try:
        visualize_long_range_correlations(corr_df_positive)
    except Exception as e:
        print("Failed Positive Test:", e)
        all_tests_passed = False
    
    # Negative Test: Missing required columns
    corr_df_negative = pd.DataFrame({
        'Feature X': ['alpha.Fz', 'beta.Cz'],
        'Feature Y': ['gamma.O1', 'delta.O2'],
        'Correlation Value': [0.8, 0.7]
    })
    try:
        visualize_long_range_correlations(corr_df_negative)
    except Exception:
        pass
    else:
        print("Failed Negative Test: Should have raised an error")
        all_tests_passed = False
    
    # Null Test: Some null values in the DataFrame
    corr_df_null = pd.DataFrame({
        'Feature 1': ['alpha.Fz', None, 'theta.Pz'],
        'Feature 2': ['gamma.O1', 'delta.O2', None],
        'Correlation': [0.8, 0.7, None]
    })
    try:
        visualize_long_range_correlations(corr_df_null)
    except Exception:
        pass
    else:
        print("Failed Null Test: Should have raised an error")
        all_tests_passed = False
    
    # Error Test: Non-DataFrame input
    try:
        visualize_long_range_correlations("Not a DataFrame")
    except Exception:
        pass
    else:
        print("Failed Error Test: Should have raised an error")
        all_tests_passed = False
    
    # Boundary Test: Minimum valid dataset (single row)
    corr_df_boundary = pd.DataFrame({
        'Feature 1': ['alpha.Fz'],
        'Feature 2': ['gamma.O1'],
        'Correlation': [0.9]
    })
    try:
        visualize_long_range_correlations(corr_df_boundary)
    except Exception as e:
        print("Failed Boundary Test:", e)
        all_tests_passed = False
    
    if all_tests_passed:
        print("visualize_long_range_correlations: Passed all tests")

if __name__ == "__main__":
    test_visualize_long_range_correlations()



def test_visualize_correlation_gradient():
    """
    Test cases for visualize_correlation_gradient to ensure handling of positive, negative, null, error, and boundary conditions.
    """
    all_tests_passed = True
    
    # Positive Test: Properly formatted EEG dataset with strong correlations
    df_positive = pd.DataFrame(
        np.random.rand(100, 5),  
        columns=['delta.Fz', 'theta.Cz', 'alpha.Pz', 'beta.O1', 'gamma.O2']
    )
    df_positive['gamma.O2'] = df_positive['beta.O1'] * 0.95 + np.random.normal(0, 0.02, size=100)  # Strong correlation
    try:
        visualize_correlation_gradient(df_positive, threshold=0.5)
    except Exception as e:
        print("Failed Positive Test:", e)
        all_tests_passed = False
    
    # Negative Test: DataFrame without EEG columns
    df_negative = pd.DataFrame({
        'random_column1': np.random.rand(100),
        'random_column2': np.random.rand(100)
    })
    try:
        visualize_correlation_gradient(df_negative, threshold=0.5)
    except Exception:
        pass
    else:
        print("Failed Negative Test: Should have raised an error")
        all_tests_passed = False
    
    # Null Test: DataFrame with all NaN values in EEG columns
    df_null = df_positive.copy()
    df_null.iloc[:, :] = np.nan  # Explicitly cast to NaN
    try:
        visualize_correlation_gradient(df_null, threshold=0.5)
    except Exception:
        pass
    else:
        print("Failed Null Test: Should have raised an error")
        all_tests_passed = False
    
    # Error Test: Non-DataFrame input
    try:
        visualize_correlation_gradient("Not a DataFrame", threshold=0.5)
    except Exception:
        pass
    else:
        print("Failed Error Test: Should have raised an error")
        all_tests_passed = False
    
    # Boundary Test: Ensure strong correlations exist for visualization
    df_boundary = pd.DataFrame(
        np.random.rand(100, 2),  # Increase sample size
        columns=['alpha.Fz', 'beta.Cz']
    )
    df_boundary['beta.Cz'] = df_boundary['alpha.Fz'] * 0.99 + np.random.normal(0, 0.01, size=100)  # Strong correlation
    try:
        visualize_correlation_gradient(df_boundary, threshold=0.5)
    except Exception as e:
        print("Failed Boundary Test:", e)
        all_tests_passed = False
    
    if all_tests_passed:
        print("visualize_correlation_gradient: Passed all tests")

if __name__ == "__main__":
    test_visualize_correlation_gradient()


def test_visualize_correlation_gradient_by_disorders():
    """
    Test cases for visualize_correlation_gradient_by_disorders to ensure handling of positive, negative, null, error, and boundary conditions.
    """
    all_tests_passed = True
    
    # Positive Test: Properly formatted EEG dataset with two disorders
    df_positive = pd.DataFrame(
        np.random.rand(100, 5),
        columns=['delta.Fz', 'theta.Cz', 'alpha.Pz', 'beta.O1', 'gamma.O2']
    )
    df_positive['main.disorder'] = np.random.choice(['Anxiety', 'Depression'], size=100)
    df_positive['gamma.O2'] = df_positive['beta.O1'] * 0.95 + np.random.normal(0, 0.02, size=100)
    try:
        visualize_correlation_gradient_by_disorders(df_positive, 'Anxiety', 'Depression', threshold=0.5)
    except Exception as e:
        print("Failed Positive Test:", e)
        all_tests_passed = False
    
    # Negative Test: DataFrame without EEG columns
    df_negative = pd.DataFrame({
        'main.disorder': np.random.choice(['Anxiety', 'Depression'], size=100),
        'random_column1': np.random.rand(100),
        'random_column2': np.random.rand(100)
    })
    try:
        visualize_correlation_gradient_by_disorders(df_negative, 'Anxiety', 'Depression', threshold=0.5)
    except Exception:
        pass
    else:
        print("Failed Negative Test: Should have raised an error")
        all_tests_passed = False
    
    # Null Test: DataFrame with all NaN values in EEG columns
    df_null = df_positive.copy()
    df_null.iloc[:, :-1] = np.nan  # Explicitly cast EEG data to NaN
    try:
        visualize_correlation_gradient_by_disorders(df_null, 'Anxiety', 'Depression', threshold=0.5)
    except Exception:
        pass
    else:
        print("Failed Null Test: Should have raised an error")
        all_tests_passed = False
    
    # Error Test: Non-DataFrame input
    try:
        visualize_correlation_gradient_by_disorders("Not a DataFrame", 'Anxiety', 'Depression', threshold=0.5)
    except Exception:
        pass
    else:
        print("Failed Error Test: Should have raised an error")
        all_tests_passed = False
    
    # Boundary Test: Ensure strong correlations exist for visualization
    df_boundary = pd.DataFrame(
        np.random.rand(100, 2),
        columns=['alpha.Fz', 'beta.Cz']
    )
    df_boundary['main.disorder'] = np.random.choice(['Anxiety', 'Depression'], size=100)
    df_boundary['beta.Cz'] = df_boundary['alpha.Fz'] * 0.99 + np.random.normal(0, 0.01, size=100)
    try:
        visualize_correlation_gradient_by_disorders(df_boundary, 'Anxiety', 'Depression', threshold=0.5)
    except Exception as e:
        print("Failed Boundary Test:", e)
        all_tests_passed = False
    
    if all_tests_passed:
        print("visualize_correlation_gradient_by_disorders: Passed all tests")

if __name__ == "__main__":
    test_visualize_correlation_gradient_by_disorders()

def test_plot_brain_activity_by_disorder():
    """
    Test cases for plot_brain_activity_by_disorder to ensure handling of positive, negative, null, error, and boundary conditions.
    """
    all_tests_passed = True
    
    # Positive Test: Properly formatted dataset with valid frequency bands
    df_positive = pd.DataFrame(
        np.random.rand(100, 6),
        columns=['delta.Fz', 'theta.Cz', 'alpha.Pz', 'beta.O1', 'gamma.O2', 'highbeta.Oz']
    )
    df_positive['main.disorder'] = np.random.choice(['Anxiety', 'Depression'], size=100)
    df_positive['specific.disorder'] = np.random.choice(['GAD', 'MDD', 'Bipolar'], size=100)
    try:
        plot_brain_activity_by_disorder(df_positive, 'Anxiety')
    except Exception as e:
        print("Failed Positive Test:", e)
        all_tests_passed = False
    
    # Negative Test: DataFrame without required columns
    df_negative = pd.DataFrame({
        'random_column1': np.random.rand(100),
        'random_column2': np.random.rand(100)
    })
    try:
        plot_brain_activity_by_disorder(df_negative, 'Anxiety')
    except Exception:
        pass
    else:
        print("Failed Negative Test: Should have raised an error")
        all_tests_passed = False
    
    # Null Test: DataFrame with all NaN values in frequency band columns
    df_null = df_positive.copy()
    df_null.iloc[:, :-2] = np.nan  # Explicitly cast EEG data to NaN
    if df_null.iloc[:, :-2].isna().all().all():
        try:
            plot_brain_activity_by_disorder(df_null, 'Anxiety')
        except Exception:
            pass
        else:
            print("Failed Null Test: Should have raised an error")
            all_tests_passed = False
    
    # Error Test: Non-DataFrame input
    try:
        plot_brain_activity_by_disorder("Not a DataFrame", 'Anxiety')
    except Exception:
        pass
    else:
        print("Failed Error Test: Should have raised an error")
        all_tests_passed = False
    
    # Boundary Test: Ensure function can handle a single disorder entry
    df_boundary = pd.DataFrame(
        np.random.rand(1, 6),
        columns=['delta.Fz', 'theta.Cz', 'alpha.Pz', 'beta.O1', 'gamma.O2', 'highbeta.Oz']
    )
    df_boundary['main.disorder'] = ['Anxiety']
    df_boundary['specific.disorder'] = ['GAD']
    try:
        plot_brain_activity_by_disorder(df_boundary, 'Anxiety')
    except Exception as e:
        print("Failed Boundary Test:", e)
        all_tests_passed = False
    
    if all_tests_passed:
        print("plot_brain_activity_by_disorder: Passed all tests")

if __name__ == "__main__":
    test_plot_brain_activity_by_disorder()