import pandas as pd
import numpy as np
from src.data_cleaning import reformat_electrode_columns
from src.data_analysis import (
    calculate_band_averages,
    prepare_disorder_band_averages,
    find_significant_differences,
    find_strong_long_range_correlations,
)


FREQUENCY_BANDS = ["delta", "theta", "alpha", "beta", "gamma", "highbeta"]
ELECTRODES = ["FP1", "FP2", "F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2"]

def get_sample_dataframe():
    return pd.DataFrame({
        "main.disorder": ["Depression", "Depression", "Healthy", "Healthy"],
        "delta.FP1": [1.2, 2.3, 1.0, 1.1],
        "theta.FP1": [2.5, 3.1, 2.4, 2.7],
        "alpha.FP1": [4.1, 3.9, 4.2, 4.0],
        "beta.FP1": [2.1, 2.0, 1.9, 2.2],
        "gamma.FP1": [3.2, 3.0, 3.1, 3.3],
        "highbeta.FP1": [1.5, 1.6, 1.4, 1.7],
    })



def test_calculate_band_averages():
    df = get_sample_dataframe()
    failed = False

    try:
        result = calculate_band_averages("Depression", df)
        assert isinstance(result, dict), f"Expected dict, but got {type(result)}"
    except Exception as e:
        print("calculate_band_averages: Positive Test Failed:", e)
        failed = True

    try:
        result = calculate_band_averages("Unknown", df)
        assert result == {}, f"Negative Test Failed: Expected {{}} but got {result}"
    except Exception as e:
        print("calculate_band_averages: Negative Test Crashed:", e)
        failed = True

    try:
        small_df = df[["main.disorder", "delta.FP1"]]
        result = calculate_band_averages("Depression", small_df)
        assert isinstance(result, dict), f"Boundary Test Failed: Expected dict but got {type(result)}"
    except Exception as e:
        print("calculate_band_averages: Boundary Test Failed:", e)
        failed = True

    try:
        df_invalid = df.copy()
        df_invalid["delta.FP1"] = ["error"] * len(df)
        result = calculate_band_averages("Depression", df_invalid)
        assert result == {}, f"Error Test Failed: Expected {{}} but got {result}"
    except Exception as e:
        print("calculate_band_averages: Error Test Crashed:", e)
        failed = True

    try:
        result = calculate_band_averages("Depression", pd.DataFrame())
        assert result == {}, f"Null Test Failed: Expected {{}} but got {result}"
    except Exception as e:
        print("calculate_band_averages: Null Test Crashed:", e)
        failed = True

    if not failed:
        print("calculate_band_averages: Passed all tests")
        
def test_prepare_disorder_band_averages():
    df = get_sample_dataframe()
    failed = False

    frequency_bands = ["delta", "theta", "alpha", "beta", "gamma", "highbeta"]
    electrodes = ["FP1", "FP2", "F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2"]
    disorders = df["main.disorder"].dropna().unique().tolist()

    try:
        result = prepare_disorder_band_averages(df, disorders, frequency_bands, electrodes)
        assert isinstance(result, tuple) and len(result) == 2, "Positive Test Failed: Expected tuple with two lists"
        assert isinstance(result[0], list) and isinstance(result[1], list), "Positive Test Failed: Expected lists inside tuple"
    except Exception as e:
        print("prepare_disorder_band_averages: Positive Test Failed:", e)
        failed = True

    try:
        result = prepare_disorder_band_averages(df, ["FakeDisorder"], frequency_bands, electrodes)
        assert result == ([], []), f"Negative Test Failed: Expected ([], []) but got {result}"
    except Exception as e:
        print("prepare_disorder_band_averages: Negative Test Crashed:", e)
        failed = True

    try:
        small_df = df.iloc[:1][["main.disorder"] + [f"{band}.{electrode}" for band in frequency_bands for electrode in electrodes if f"{band}.{electrode}" in df.columns]]
        result = prepare_disorder_band_averages(small_df, disorders, frequency_bands, electrodes)
        assert isinstance(result, tuple), "Boundary Test Failed: Expected a tuple"
    except Exception as e:
        print("prepare_disorder_band_averages: Boundary Test Failed:", e)
        failed = True

    try:
        df_invalid = df.copy()
        for band in frequency_bands:
            df_invalid[f"{band}.FP1"] = ["error"] * len(df)
        result = prepare_disorder_band_averages(df_invalid, disorders, frequency_bands, electrodes)
        assert isinstance(result, tuple), f"Error Test Failed: Expected a tuple but got {result}"
    except Exception as e:
        print("prepare_disorder_band_averages: Error Test Crashed:", e)
        failed = True

    try:
        result = prepare_disorder_band_averages(pd.DataFrame(), disorders, frequency_bands, electrodes)
        assert result == ([], []), f"Null Test Failed: Expected ([], []) but got {result}"
    except Exception as e:
        print("prepare_disorder_band_averages: Null Test Crashed:", e)
        failed = True

    if not failed:
        print("prepare_disorder_band_averages: Passed all tests")

def test_find_significant_differences():
    df = get_sample_dataframe()
    failed = False

    try:
        result = find_significant_differences(df, "Depression", "Healthy", p_threshold=0.1)
        assert isinstance(result, dict), f"Expected dict, but got {type(result)}"
    except Exception as e:
        print("find_significant_differences: Positive Test Failed:", e)
        failed = True

    try:
        result = find_significant_differences(df, "FakeDisorder", "Healthy")
        assert result == {}, f"Negative Test Failed: Expected {{}} but got {result}"
    except Exception as e:
        print("find_significant_differences: Negative Test Crashed:", e)
        failed = True

    try:
        small_df = df[["main.disorder", "delta.FP1"]]
        result = find_significant_differences(small_df, "Depression", "Healthy")
        assert isinstance(result, dict), f"Boundary Test Failed: Expected dict but got {type(result)}"
    except Exception as e:
        print("find_significant_differences: Boundary Test Failed:", e)
        failed = True

    try:
        df_invalid = df.copy()
        df_invalid["delta.FP1"] = ["error"] * len(df)
        result = find_significant_differences(df_invalid, "Depression", "Healthy")
        assert result == {}, f"Error Test Failed: Expected {{}} but got {result}"
    except Exception as e:
        print("find_significant_differences: Error Test Crashed:", e)
        failed = True

    try:
        result = find_significant_differences(pd.DataFrame(), "Depression", "Healthy")
        assert result == {}, f"Null Test Failed: Expected {{}} but got {result}"
    except Exception as e:
        print("find_significant_differences: Null Test Crashed:", e)
        failed = True

    if not failed:
        print("find_significant_differences: Passed all tests")


def test_find_strong_long_range_correlations():
    df = get_sample_dataframe()
    failed = False

    try:
        result = find_strong_long_range_correlations(df, num_pairs=3, threshold=0.5)
        assert isinstance(result, pd.DataFrame), f"Expected DataFrame, but got {type(result)}"
    except Exception as e:
        print("find_strong_long_range_correlations: Positive Test Failed:", e)
        failed = True

    try:
        result = find_strong_long_range_correlations(df, num_pairs=3, threshold=2)
        assert result.empty, f"Negative Test Failed: Expected empty DataFrame but got {result}"
    except Exception as e:
        print("find_strong_long_range_correlations: Negative Test Crashed:", e)
        failed = True

    try:
        result = find_strong_long_range_correlations(df.iloc[:1], num_pairs=3, threshold=0.5)
        assert isinstance(result, pd.DataFrame), f"Boundary Test Failed: Expected DataFrame but got {type(result)}"
    except Exception as e:
        print("find_strong_long_range_correlations: Boundary Test Failed:", e)
        failed = True

    try:
        df_invalid = df.copy()
        df_invalid["delta.FP1"] = ["error"] * len(df)
        result = find_strong_long_range_correlations(df_invalid)
        assert result.empty, f"Error Test Failed: Expected empty DataFrame but got {result}"
    except Exception as e:
        print("find_strong_long_range_correlations: Error Test Crashed:", e)
        failed = True

    try:
        result = find_strong_long_range_correlations(pd.DataFrame())
        assert result.empty, f"Null Test Failed: Expected empty DataFrame but got {result}"
    except Exception as e:
        print("find_strong_long_range_correlations: Null Test Crashed:", e)
        failed = True

    if not failed:
        print("find_strong_long_range_correlations: Passed all tests")


if __name__ == "__main__":
    test_calculate_band_averages()
    test_find_significant_differences()
    test_find_strong_long_range_correlations()
    test_prepare_disorder_band_averages()