from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):
    """
    Preprocesses the EEG dataset:
    - Drops unnecessary columns
    - Encodes categorical variables
    - Returns the processed dataframe and label encoders
    """
    df = df.drop(columns=["no.", "eeg.date", "specific.disorder"])
    
    label_encoders = {}
    for col in ["sex", "main.disorder"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Save encoder for later interpretation

    return df, label_encoders

def split_data(df):
    """
    Splits the dataset into training and test sets.
    Returns X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=["main.disorder"])
    y = df["main.disorder"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """
    Trains a Random Forest model on the given training data.
    Returns the trained model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluates the trained model on the test set.
    Returns accuracy and classification report as a dictionary.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Convert classification report to a dictionary instead of a string
    classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    
    return accuracy, classification_rep
