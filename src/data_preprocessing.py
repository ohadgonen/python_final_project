import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# Function to select only EEG columns
def select_eeg_columns(df):
    eeg_columns = [col for col in df.columns if col.startswith('AB.A.') or col.startswith('COH.F.')]
    df_eeg = df[eeg_columns]
    return df_eeg

# Function to preprocess the target column (convert from string labels to numeric)
def preprocess_target(y):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    return y


from sklearn.model_selection import train_test_split

# Function to split the data into training and testing sets
def split_data(df, target_column, test_size=0.2):
    # Select only the EEG columns for input features
    X = select_eeg_columns(df)

    # Extract the target column (main.disorder) and preprocess it
    y = df[target_column]
    y = preprocess_target(y)

    # Split the data into training and testing sets (both X and y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

from sklearn.linear_model import LogisticRegression

# Function to train the logistic regression model
def train_model(X_train, y_train):
    # Initialize the logistic regression model
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    return model

from sklearn.metrics import accuracy_score, classification_report

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Display the classification report (precision, recall, F1-score)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

