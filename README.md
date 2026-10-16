# Bearing Fault Detection using Deep Learning

A deep learning-based predictive maintenance system that detects bearing faults from vibration signals using a **1D Convolutional Autoencoder**. The model is trained only on healthy bearing data and identifies faulty bearings by measuring reconstruction error.

---

## Overview

Bearings are one of the most critical components in rotating machinery. Faulty bearings can lead to unexpected equipment failure, costly downtime, and reduced operational efficiency.

This project uses an **unsupervised anomaly detection** approach:

- Train the autoencoder using only healthy bearing vibration signals.
- Learn the normal operating characteristics of the bearing.
- Detect faults by identifying signals that cannot be reconstructed accurately.

Unlike supervised classifiers, this approach does not require fault labels during training, making it suitable for real-world predictive maintenance applications.

---

## Features

- Vibration signal preprocessing
- Fast Fourier Transform (FFT) for frequency-domain analysis
- Sliding window segmentation
- Conv1D Autoencoder architecture
- Reconstruction error-based anomaly detection
- Automatic threshold selection from healthy data
- Performance evaluation on healthy and faulty datasets

---

## Project Workflow

```
Raw Bearing Signal
        │
        ▼
Signal Preprocessing
        │
        ▼
Fast Fourier Transform (FFT)
        │
        ▼
Frequency Spectrum
        │
        ▼
Sliding Window Generation
        │
        ▼
Conv1D Autoencoder
        │
        ▼
Reconstructed Signal
        │
        ▼
Reconstruction Error
        │
        ▼
Threshold Comparison
        │
        ▼
Healthy / Faulty
```

---

## Model Architecture

Encoder

- Conv1D
- ReLU
- MaxPooling1D
- Conv1D
- ReLU
- MaxPooling1D

Latent Representation

Decoder

- Conv1D
- UpSampling1D
- Conv1D
- UpSampling1D
- Conv1D (Linear Output)

Loss Function

```
Mean Squared Error (MSE)
```

Optimizer

```
Adam
```

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Dataset

The model is trained using healthy bearing vibration signals and evaluated on both healthy and faulty samples.

The workflow includes:

- Healthy Data → Training
- Healthy Test Data → Evaluation
- Faulty Data → Evaluation

---

## Threshold Selection

The anomaly threshold is determined using reconstruction errors from healthy training data.

```
Threshold = Maximum Reconstruction Error on Healthy Data
```

If

```
Reconstruction Error > Threshold
```

the bearing is classified as **Faulty**.

Otherwise,

```
Healthy
```

---

## Evaluation

The model is evaluated using:

- Accuracy
- Confusion Matrix
- Reconstruction Error

---

## Repository Structure

```
Bearing-Fault-Detection/
│
├── notebook.ipynb
├── README.md
├── requirements.txt
├── dataset/
└── model/
```

---

## Future Improvements

- Streamlit Web Application
- Real-time vibration monitoring
- Model deployment
- Interactive visualization dashboard
- Support for multiple bearing fault categories
- Cloud deployment

---

## Learning Outcomes

This project helped me understand:

- Signal Processing
- Fast Fourier Transform (FFT)
- Frequency-domain feature extraction
- Convolutional Neural Networks
- Autoencoders
- Anomaly Detection
- Deep Learning using TensorFlow/Keras
- Model evaluation and threshold-based classification

---

## Author

**Sricharan Pusarla**

VIT Vellore

Interested in Artificial Intelligence, Machine Learning, and Predictive Maintenance.

---