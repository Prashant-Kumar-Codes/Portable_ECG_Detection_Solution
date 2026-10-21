"""
inference.py
------------
Model inference engine for real-time ECG arrhythmia detection.

Loads a trained ECGCNNBiLSTM checkpoint and exposes InferenceEngine.predict()
which preprocesses a 2500-sample raw window and returns the predicted class,
confidence score, and full class probability distribution.

Notes
-----
- source_fs is always 250 Hz for live ESP32 data (no resampling needed).
- The same preprocess_ecg_window used during training is applied here
  identically (see preprocessing.py).
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from config import BEST_MODEL_PATH, CLASS_NAMES
from model import ECGCNNBiLSTM, build_model
from preprocessing import preprocess_ecg_window

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Load a trained model checkpoint and run window-level arrhythmia inference.

    Parameters
    ----------
    model_path : Path, optional
        Path to the .pt checkpoint file. Defaults to config.BEST_MODEL_PATH.
    device : str, optional
        PyTorch device string ('cpu', 'cuda', 'cuda:0'). Defaults to 'cpu'.
    """

    def __init__(self, model_path: Path = BEST_MODEL_PATH, device: str = "cpu"):
        self.device = torch.device(device)
        self.model  = self._load_model(model_path)

    def _load_model(self, model_path: Path) -> ECGCNNBiLSTM:
        """Load the model state dict from a checkpoint file.

        Supports both plain state_dict saves and Trainer-style checkpoints
        {'model_state_dict': ..., 'epoch': ...}.
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at: {model_path}\n"
                "Run Notebooks/N5_Training.ipynb first to train and save the model."
            )
        model = build_model(str(self.device))
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"Model loaded from '{model_path}' on device '{self.device}'.")
        return model

    @torch.no_grad()
    def predict(self, raw_window: np.ndarray, lead_off: bool = False) -> dict:
        """Classify a single 2500-sample ECG window.

        Parameters
        ----------
        raw_window : np.ndarray
            1-D array of 2500 raw samples from the ESP32 AD8232 ADC (0-4095)
            or float mV values. Z-score normalisation bridges the scale gap.
        lead_off : bool, optional
            True if electrodes are disconnected (AD8232 LO+ / LO- active).

        Returns
        -------
        dict
            class_id      : int   - predicted class index (0-4)
            class_name    : str   - human-readable label
            confidence    : float - softmax probability of predicted class
            probabilities : list  - softmax probs for all 5 classes
            lead_off      : bool  - whether lead-off was triggered
        """
        if lead_off:
            return {
                "class_id":      4,
                "class_name":    "Noise / Artifact (Lead-Off)",
                "confidence":    1.0,
                "probabilities": [0.0, 0.0, 0.0, 0.0, 1.0],
                "lead_off":      True,
            }

        # source_fs=250: ESP32 already streams at target rate, no resampling
        x = preprocess_ecg_window(raw_window, source_fs=250, target_fs=250)

        # Shape: (1, 1, 2500)
        tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(self.device)

        logits = self.model(tensor)                                  # (1, 5)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (5,)

        class_id   = int(probs.argmax())
        confidence = float(probs[class_id])

        return {
            "class_id":      class_id,
            "class_name":    CLASS_NAMES[class_id],
            "confidence":    round(confidence, 4),
            "probabilities": [round(float(p), 4) for p in probs],
            "lead_off":      False,
        }
