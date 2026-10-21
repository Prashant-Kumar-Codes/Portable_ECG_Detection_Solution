"""
preprocessing.py
----------------
Shared ECG signal preprocessing pipeline.

The function `preprocess_ecg_window` is the single canonical preprocessing
step used IDENTICALLY in:
  - Notebooks/N2_Preprocessing.ipynb (development & validation)
  - Modules/dataset_builder.py       (training data construction)
  - Inference server                 (live ESP32 window processing)

Never modify this function's behaviour without updating all three call-sites.
"""

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

from config import (
    SOURCE_FS, TARGET_FS, RESAMPLE_UP, RESAMPLE_DOWN,
    FILTER_LOWCUT, FILTER_HIGHCUT, FILTER_ORDER, WINDOW_SAMPLES,
)


def _butter_bandpass_coeffs(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int,
) -> tuple:
    """Return (b, a) Butterworth bandpass filter coefficients.

    Parameters
    ----------
    lowcut : float
        Low cut-off frequency in Hz.
    highcut : float
        High cut-off frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order.

    Returns
    -------
    b, a : ndarray
        IIR filter numerator and denominator polynomials.
    """
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


# Pre-compute filter coefficients at import time (constants never change)
_B, _A = _butter_bandpass_coeffs(FILTER_LOWCUT, FILTER_HIGHCUT, TARGET_FS, FILTER_ORDER)


def preprocess_ecg_window(
    window: np.ndarray,
    source_fs: int = SOURCE_FS,
    target_fs: int = TARGET_FS,
) -> np.ndarray:
    """Apply the canonical ECG preprocessing pipeline to a single window.

    Steps (must match inference pipeline exactly):
      1. Replace NaN / Inf with 0.0.
      2. Resample from source_fs to target_fs using resample_poly (integer ratio).
      3. Apply zero-phase 4th-order Butterworth bandpass (0.5–40 Hz) via filtfilt.
      4. Per-window Z-score normalisation (mean=0, std≈1).

    Parameters
    ----------
    window : np.ndarray
        1-D float array of raw ECG samples at `source_fs` Hz.
        Expected shape: (source_fs * WINDOW_SEC,) e.g. (5000,) for 500 Hz.
    source_fs : int, optional
        Sampling rate of the input signal. Defaults to SOURCE_FS (500 Hz).
    target_fs : int, optional
        Target sampling rate after resampling. Defaults to TARGET_FS (250 Hz).

    Returns
    -------
    np.ndarray
        Preprocessed 1-D float32 array of length `WINDOW_SAMPLES` (2500).

    Raises
    ------
    ValueError
        If the window is too short to filter after resampling (<= filter order * 3).
    """
    # Step 1: Sanitise — replace NaN / Inf
    x = np.where(np.isfinite(window), window, 0.0)

    # Step 2: Resample if needed
    if source_fs != target_fs:
        x = resample_poly(x, up=RESAMPLE_UP, down=RESAMPLE_DOWN)

    # Truncate or zero-pad to exactly WINDOW_SAMPLES
    if len(x) > WINDOW_SAMPLES:
        x = x[:WINDOW_SAMPLES]
    elif len(x) < WINDOW_SAMPLES:
        pad = np.zeros(WINDOW_SAMPLES - len(x), dtype=x.dtype)
        x = np.concatenate([x, pad])

    # Step 3: Bandpass filter (zero-phase)
    min_len = FILTER_ORDER * 3 + 1
    if len(x) <= min_len:
        raise ValueError(
            f"Window too short for filtering: got {len(x)} samples, need >{min_len}."
        )
    x = filtfilt(_B, _A, x)

    # Step 4: Z-score normalisation (epsilon prevents division by zero on flat lines)
    mean = x.mean()
    std  = x.std()
    x = (x - mean) / (std + 1e-6)

    return x.astype(np.float32)


def is_valid_window(raw_window: np.ndarray, min_valid_frac: float = 0.5) -> bool:
    """Return True if the window has enough non-NaN samples to be usable.

    Parameters
    ----------
    raw_window : np.ndarray
        Raw signal array (may contain NaN).
    min_valid_frac : float, optional
        Minimum fraction of non-NaN samples required. Default 0.5.

    Returns
    -------
    bool
    """
    valid_count = np.sum(np.isfinite(raw_window))
    return (valid_count / len(raw_window)) >= min_valid_frac
