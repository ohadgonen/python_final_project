# EEG Psychiatric Descriptions

## Description

This project analyzes EEG data for psychiatric disorder classification. It includes data processing, visualization, and machine learning models to classify different psychiatric disorders based on EEG signals.

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

Example usage in a Jupyter Notebook:
```python
from src.data_visualization import visualize_average_activity

# Assuming eeg_data is already created
visualize_average_activity(eeg_data, 'addictive disorder')


# Dependencies

numpy==2.1.3
pandas==2.2.3
scikit-learn==1.6.0
torch==2.5.1
matplotlib==3.9.2
scipy==1.14.0
seaborn==0.12.2
mne==1.2.2
networkx==3.0
imbalanced-learn==0.10.1
