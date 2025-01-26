import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

from src.model import preprocess_data, split_data, train_model, evaluate_model  

def get_sample_dataframe():
    """Creates a small sample EEG dataset for testing."""
    data = {
        "no.": [1, 2, 3, 4, 5],
        "eeg.date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
        "sex": ["M", "F", "M", "F", "M"],
        "education": [12, 16, 14, 12, 18],
        "IQ": [110, 105, 120, 115, 98],
        "main.disorder": ["Depression", "Anxiety", "Depression", "Bipolar", "Anxiety"],
        "specific.disorder": ["Major", "GAD", "Major", "BP1", "GAD"],
        "delta.FP1": [0.5, 0.8, 0.6, 0.9, 0.7],
        "theta.FP1": [0.3, 0.4, 0.5, 0.6, 0.7],
        "alpha.FP1": [0.7, 0.5, 0.6, 0.8, 0.9],
        "beta.FP1": [0.2, 0.3, 0.4, 0.5, 0.6],
        "gamma.FP1": [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    return pd.DataFrame(data)


def test_preprocess_data():
    df = get_sample_dataframe()
    failed = False

    try:
        processed_df, encoders = preprocess_data(df)
        assert "no." not in processed_df.columns, "Preprocessing Test Failed: 'no.' column not dropped"
        assert "eeg.date" not in processed_df.columns, "Preprocessing Test Failed: 'eeg.date' column not dropped"
        assert "specific.disorder" not in processed_df.columns, "Preprocessing Test Failed: 'specific.disorder' column not dropped"
        assert "sex" in processed_df.columns and isinstance(processed_df["sex"].dtype, np.dtype), "Preprocessing Test Failed: 'sex' not encoded"
        assert "main.disorder" in processed_df.columns and isinstance(processed_df["main.disorder"].dtype, np.dtype), "Preprocessing Test Failed: 'main.disorder' not encoded"
    except Exception as e:
        print("Preprocessing Test Failed:", e)
        failed = True

    if not failed:
        print("Preprocessing Test Passed")


def test_split_data():
    df = get_sample_dataframe()
    failed = False

    try:
        processed_df, _ = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(processed_df)
        assert len(X_train) > 0 and len(X_test) > 0, "Splitting Test Failed: X_train or X_test is empty"
        assert len(y_train) > 0 and len(y_test) > 0, "Splitting Test Failed: y_train or y_test is empty"
        assert len(X_train) > len(X_test), "Splitting Test Failed: Incorrect train-test split ratio"
    except Exception as e:
        print("Splitting Test Failed:", e)
        failed = True

    if not failed:
        print("Splitting Test Passed")


def test_train_model():
    df = get_sample_dataframe()
    failed = False

    try:
        processed_df, _ = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(processed_df)
        model = train_model(X_train, y_train)
        assert isinstance(model, RandomForestClassifier), "Training Test Failed: Model is not a RandomForestClassifier"
        assert hasattr(model, "predict"), "Training Test Failed: Model does not have predict method"
    except Exception as e:
        print("Training Test Failed:", e)
        failed = True

    if not failed:
        print("Training Test Passed")


def test_evaluate_model():
    df = get_sample_dataframe()
    failed = False

    try:
        processed_df, encoders = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(processed_df)
        model = train_model(X_train, y_train)
        accuracy, report = evaluate_model(model, X_test, y_test, encoders["main.disorder"])

        assert isinstance(accuracy, float), "Evaluation Test Failed: Accuracy is not a float"
        assert isinstance(report, dict), "Evaluation Test Failed: Report is not a dictionary"
        assert "Depression" in report, "Evaluation Test Failed: Classification report missing expected disorder labels"
    except Exception as e:
        print("Evaluation Test Failed:", e)
        failed = True

    if not failed:
        print("Evaluation Test Passed")


def run_all_tests():
    test_preprocess_data()
    test_split_data()
    test_train_model()
    test_evaluate_model()


if __name__ == "__main__":
    run_all_tests()
