from sklearn.model_selection import train_test_split
# can we predict a patient's main psychiatric disoder based on their EEG data?
# Function to split the data into training and testing sets
def split_data(df, target_column, test_size=0.2):
    # Select only the EEG columns for input features
    X = select_eeg_columns(df)

    # Extract the target column (main.disorder) and preprocess it
    y = df[target_column]
    y = preprocess_target(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test
