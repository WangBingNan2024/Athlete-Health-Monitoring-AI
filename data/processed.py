import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from scipy import signal
import matplotlib.pyplot as plt


def load_eeg_data(file_path):
    """Load EEG data from a file (e.g., .edf, .bdf)"""
    raw_data = mne.io.read_raw_edf(file_path, preload=True)
    return raw_data


def preprocess_eeg_data(raw_data):
    """Preprocess EEG data: filtering, baseline correction, and artifact removal."""

    # Apply bandpass filter (1-40 Hz) to remove unwanted frequencies
    raw_data.filter(1, 40, fir_design='firwin')

    # Perform ICA to detect and remove eye blink artifacts
    ica = ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw_data)
    ica.exclude = []  # Mark the components to exclude after visual inspection
    raw_data = ica.apply(raw_data)

    # Baseline correction for each epoch
    raw_data = raw_data.copy().crop(tmin=0, tmax=raw_data.times[-1])
    raw_data.apply_baseline(baseline=(None, 0))  # Apply baseline correction to the full signal

    return raw_data


def segment_eeg_data(raw_data, event_id, tmin, tmax):
    """Segment the EEG data based on event markers."""
    events = mne.find_events(raw_data)
    epochs = mne.Epochs(raw_data, events, event_id, tmin, tmax, baseline=None, detrend=1, picks='eeg')
    return epochs


def extract_features(epochs):
    """Extract time-domain and frequency-domain features from EEG epochs."""

    # Time-domain features
    mean_amplitude = np.mean(epochs.get_data(), axis=2)
    variance = np.var(epochs.get_data(), axis=2)

    # Frequency-domain features using Fast Fourier Transform (FFT)
    n_channels, n_times = epochs.get_data().shape[1], epochs.get_data().shape[2]
    psd, freqs = signal.welch(epochs.get_data(), fs=epochs.info['sfreq'], nperseg=n_times // 2)

    # Extract features in specific frequency bands (e.g., Alpha, Beta, Gamma, etc.)
    alpha_band = (8, 13)
    beta_band = (13, 30)
    gamma_band = (30, 40)

    alpha_psd = np.mean(psd[:, :, (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])], axis=2)
    beta_psd = np.mean(psd[:, :, (freqs >= beta_band[0]) & (freqs <= beta_band[1])], axis=2)
    gamma_psd = np.mean(psd[:, :, (freqs >= gamma_band[0]) & (freqs <= gamma_band[1])], axis=2)

    # Collect the features in a dataframe
    features = pd.DataFrame({
        'mean_amplitude': np.mean(mean_amplitude, axis=1),
        'variance': np.mean(variance, axis=1),
        'alpha_psd': np.mean(alpha_psd, axis=1),
        'beta_psd': np.mean(beta_psd, axis=1),
        'gamma_psd': np.mean(gamma_psd, axis=1)
    })

    return features


def plot_eeg_data(epochs):
    """Visualize some of the EEG epochs."""
    epochs.plot(n_epochs=5, duration=10, title="EEG Epochs")


# Example usage
file_path = 'path_to_your_eeg_data_file.edf'  # Replace with actual path to your EEG data file
raw_data = load_eeg_data(file_path)
preprocessed_data = preprocess_eeg_data(raw_data)

# Segment the EEG data based on specific event markers
event_id = 1  # Example event ID, change according to your dataset
epochs = segment_eeg_data(preprocessed_data, event_id, tmin=-0.5, tmax=2.0)

# Extract features from the EEG epochs
features = extract_features(epochs)
print(features.head())

# Plot the EEG data
plot_eeg_data(epochs)
