import os
import numpy as np
from scipy.io import loadmat
import tensorflow as tf

def evaluate_signal(mat_file_path, model_path="bearing_autoencoder.keras", threshold=2.8772, window_size=1024):
    """
    Evaluates vibration signal from a .mat file for bearing anomalies.
    """
    if not os.path.exists(mat_file_path):
        print(f"File not found: {mat_file_path}")
        return

    try:
        data = loadmat(mat_file_path)
    except Exception as e:
        print(f"Error loading {mat_file_path}: {e}")
        return

    # Find the vibration signal key (DE: Drive End)
    signal_key = None
    for k in data.keys():
        if "DE_time" in k:
            signal_key = k
            break
    if not signal_key:
        keys = [k for k in data.keys() if not k.startswith("__")]
        signal_key = keys[0] if keys else None

    if not signal_key:
        print(f"Could not find vibration signal in {mat_file_path}")
        return

    signal = data[signal_key].flatten()
    print(f"Loaded signal '{signal_key}' with {len(signal)} data points from {mat_file_path}")

    # Sliding window segmentation
    windows = []
    for i in range(0, len(signal), window_size):
        chunk = signal[i:i + window_size]
        if len(chunk) == window_size:
            windows.append(chunk)

    if not windows:
        print("Signal is shorter than window_size.")
        return

    windows = np.array(windows)

    # FFT and take positive half spectrum
    fft_windows = np.abs(np.fft.fft(windows, axis=1))
    half_spec = fft_windows[:, :window_size // 2]
    inputs = half_spec.reshape(-1, window_size // 2, 1)

    # Load model and predict
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Please train and save the model first.")
        return

    model = tf.keras.models.load_model(model_path)
    reconstructions = model.predict(inputs)
    errors = ((inputs - reconstructions) ** 2).mean(axis=(1, 2))

    anomalies = errors > threshold
    num_anomalies = np.sum(anomalies)
    total_windows = len(anomalies)

    print("\n--- Diagnostic Results ---")
    print(f"Total Windows: {total_windows}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Mean Reconstruction Error: {errors.mean():.4f}")
    print(f"Max Reconstruction Error: {errors.max():.4f}")
    print(f"Anomalous Windows: {num_anomalies} / {total_windows} ({num_anomalies / total_windows * 100:.1f}%)")

    if num_anomalies > total_windows * 0.5:
        print("STATUS: FAULTY (Anomaly Detected)")
    else:
        print("STATUS: HEALTHY (Normal Operation)")

if __name__ == "__main__":
    # Test on default files if present
    for test_file in ["normal.mat", "faulty.mat"]:
        if os.path.exists(test_file):
            print(f"\n================ Testing {test_file} ================")
            evaluate_signal(test_file)