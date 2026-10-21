# Model Design Rationale

## Why 1D CNN + BiLSTM?

### 1D CNN Front-End
ECG morphology (P-wave shape, QRS width, T-wave inversion) is a local, short-range pattern.
1D convolutions with progressively larger receptive fields extract these efficiently.
4 blocks reduce L from 2500 to 78, compressing 32x while preserving morphological tokens.

### BiLSTM Back-End
Rhythm classification (AFib vs. Normal) depends on the regularity of R-R intervals across
the full 10-second window - a long-range dependency. BiLSTM processes the 78 CNN tokens in
both directions, capturing irregular rhythm patterns detectable forward and backward.

### Why not Transformer?
Student hardware constraint. BiLSTM on 78 tokens trains comfortably on a single consumer GPU
(4-8 GB VRAM). A Transformer encoder would work similarly but adds ~3x parameters for
marginal gain on a 10-second single-lead task.

## Class Imbalance Handling
Class weights computed via balanced inverse-frequency weighting:
`weight_k = n_samples / (n_active_classes * count_k)`
Any class with count == 0 is assigned a neutral weight of 1.0 to prevent division by zero or errors.
Passed to CrossEntropyLoss(weight=...) to up-weight rare classes (VT, SVA).
Avoid oversampling - data leakage risk with windowed time series.

## Input Normalization
Per-window Z-score (NOT global dataset statistics) because:
1. Bridges scale gap between VitalDB mV floats and ESP32 ADC integer output
2. Makes model invariant to electrode impedance variation between patients
3. Computed fresh on every inference window - no stored mean/std artifact needed
