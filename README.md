# EEG Psychiatric Descriptions

## Description

This project analyzes EEG data for psychiatric disorder classification. It includes data processing, visualization, and a machine learning model to classify different psychiatric disorders based on EEG signals.


## Data source

the data set was found in kaggle, under the name 'EEG Psychiatric Disorders Dataset'.


Link : https://www.kaggle.com/datasets/shashwatwork/eeg-psychiatric-disorders-dataset?select=EEG.machinelearing_data_BRMH.csv

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/eeg-psychiatric-descriptions.git
    cd eeg-psychiatric-descriptions
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your EEG data and ensure it follows the required format.
2. Run the Jupyter Notebook `main.ipynb` to perform data analysis and visualization.
3. Use the provided functions in `src/data_visualization.py` to visualize average electrode activity for specific disorders.

## Testing 

This project uses pytest for testing.

pytest installation: 

Before running tests, ensure pytest is installed. You can install it using:
```sh
pip install pytest
```
If you’re using a virtual environment (.venv), activate it first:
```sh
source .venv/bin/activate  # On macOS/Linux
```
```sh
.\.venv\Scripts\activate   # On Windows (PowerShell)
```

Then install pytest:
```sh
pip install pytest
```

Running tests: 
to run each test module, simply write pytest test_module.py

for example:
```sh
pytest test_data_analysis.py
```

# Dependencies

numpy==2.1.3

pandas==2.2.3

scikit-learn==1.6.0

torch==2.5.1

matplotlib==3.9.2

scipy==1.14.0

seaborn==0.12.

mne==1.2.2

networkx==3.0

imbalanced-learn==0.10.1
