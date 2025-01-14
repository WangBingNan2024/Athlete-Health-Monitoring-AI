import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(data_path):
    """Load and preprocess the data from the given path."""
    # Example loading raw EEG, heart rate, and movement data
    eeg_data = pd.read_csv(os.path.join(data_path, 'eeg_data.csv'))
    hr_data = pd.read_csv(os.path.join(data_path, 'hr_data.csv'))
    movement_data = pd.read_csv(os.path.join(data_path, 'movement_data.csv'))
    labels = pd.read_csv(os.path.join(data_path, 'labels.csv'))

    return eeg_data, hr_data, movement_data, labels


def preprocess_data(eeg_data, hr_data, movement_data):
    """Preprocess the data: scaling and splitting into training and test sets."""
    scaler = StandardScaler()

    # Scaling each modality (EEG, HR, Movement)
    eeg_data_scaled = scaler.fit_transform(eeg_data)
    hr_data_scaled = scaler.fit_transform(hr_data)
    movement_data_scaled = scaler.fit_transform(movement_data)

    # Split data into train and test sets (80-20 split)
    X_eeg_train, X_eeg_test, X_hr_train, X_hr_test, X_movement_train, X_movement_test, y_train, y_test = train_test_split(
        eeg_data_scaled, hr_data_scaled, movement_data_scaled, labels, test_size=0.2, random_state=42
    )

    return X_eeg_train, X_eeg_test, X_hr_train, X_hr_test, X_movement_train, X_movement_test, y_train, y_test


def create_tensor_datasets(X_eeg_train, X_eeg_test, X_hr_train, X_hr_test, X_movement_train, X_movement_test, y_train,
                           y_test):
    """Convert the numpy arrays to PyTorch tensors."""
    train_data = (torch.tensor(X_eeg_train, dtype=torch.float32),
                  torch.tensor(X_hr_train, dtype=torch.float32),
                  torch.tensor(X_movement_train, dtype=torch.float32),
                  torch.tensor(y_train, dtype=torch.float32))

    test_data = (torch.tensor(X_eeg_test, dtype=torch.float32),
                 torch.tensor(X_hr_test, dtype=torch.float32),
                 torch.tensor(X_movement_test, dtype=torch.float32),
                 torch.tensor(y_test, dtype=torch.float32))

    return train_data, test_data


# Example usage
if __name__ == "__main__":
    data_path = "path_to_data_folder"  # Replace with actual path
    eeg_data, hr_data, movement_data, labels = load_data(data_path)
    X_eeg_train, X_eeg_test, X_hr_train, X_hr_test, X_movement_train, X_movement_test, y_train, y_test = preprocess_data(
        eeg_data, hr_data, movement_data)
    train_data, test_data = create_tensor_datasets(X_eeg_train, X_eeg_test, X_hr_train, X_hr_test, X_movement_train,
                                                   X_movement_test, y_train, y_test)
    print("Data preprocessing complete!")
