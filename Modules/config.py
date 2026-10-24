"""
config.py
---------
Central configuration for the AI-Based Single-Lead ECG Arrhythmia Detection pipeline.
All constants used across Modules/ and Notebooks/ are defined here.
Import this module everywhere; never hardcode values elsewhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT     = Path(__file__).resolve().parent.parent
DATASET_ROOT     = PROJECT_ROOT / "Dataset" / "ecg_1_lead_dataset"
ANNOTATION_DIR   = DATASET_ROOT / "Annotation_Files"
METADATA_CSV     = DATASET_ROOT / "metadata.csv"
OUTPUT_DATA_DIR  = PROJECT_ROOT / "Dataset" / "Output_Data"
SAVED_MODELS_DIR = OUTPUT_DATA_DIR / "saved_models"

# Ensure output directories exist at import time
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Signal constants
# ---------------------------------------------------------------------------
SOURCE_FS       = 500        # Hz — VitalDB native sampling rate
TARGET_FS       = 250        # Hz — target (matches ESP32 hardware)
RESAMPLE_UP     = 1          # resample_poly numerator   (500->250 is 1:2)
RESAMPLE_DOWN   = 2          # resample_poly denominator

WINDOW_SEC      = 5         # seconds per classification window
WINDOW_SAMPLES  = TARGET_FS * WINDOW_SEC   # 2500 samples
STRIDE_SAMPLES  = TARGET_FS * 2            # 500 samples = 2-second stride

# Butterworth bandpass filter parameters
FILTER_LOWCUT   = 0.5        # Hz — removes baseline wander
FILTER_HIGHCUT  = 40.0       # Hz — removes high-freq noise / powerline artifacts
FILTER_ORDER    = 4

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------
NUM_CLASSES = 5

# Integer label → human-readable name
CLASS_NAMES = {
    0: "Normal Sinus Rhythm",
    1: "Atrial Fibrillation",
    2: "Supraventricular Arrhythmia",
    3: "Ventricular Arrhythmia",
    4: "Noise / Artifact",
}

# Exact rhythm_label strings from Annotation_Files CSVs → integer class
# Priority: bad_signal_quality==True or rhythm_label=="Noise" → class 4 first
RHYTHM_TO_CLASS = {
    # Class 0: Normal Sinus Rhythm
    "N":                                                0,
    "Normal Sinus Rhythm":                              0,
    # Class 1: Atrial Fibrillation / Flutter
    "AFIB/AFL":                                         1,
    "Atrial Fibrillation":                              1,
    "Atrial fibrillation":                              1,
    # Class 2: Supraventricular Arrhythmia
    "SVTA":                                             2,
    "Supraventricular Tachyarrhythmia":                 2,
    "Patterned Atrial Ectopy":                          2,
    "WAP/MAT":                                          2,
    "Wandering Atrial Pacemaker / Multifocal Atrial Rhythm": 2,
    "SND":                                              2,
    "Sinus Node Dysfunction":                           2,
    "AVB":                                              2,
    "Atrioventricular Block":                           2,
    # Class 3: Ventricular Arrhythmia
    "VT":                                               3,
    "Ventricular Tachyarrhythmia":                      3,
    "Patterned Ventricular Ectopy":                     3,
    # Class 4: Noise / Artifact
    "Noise":                                            4,
    "Unclassifiable":                                   4,   # treat as noise
}

# ---------------------------------------------------------------------------
# Dataset split fractions (applied patient-wise on case_id)
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
TEST_FRAC  = 0.10
SPLIT_SEED = 42

# Minimum fraction of non-NaN samples required in a window to keep it
MIN_VALID_FRAC = 0.50

# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------
LSTM_HIDDEN_SIZE = 164
LSTM_NUM_LAYERS  = 3
LSTM_DROPOUT     = 0.3
FC_HIDDEN_SIZE   = 78
CLASSIFIER_DROPOUT = 0.3

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE      = 64
LEARNING_RATE   = 1e-3
MAX_EPOCHS      = 50
EARLY_STOP_PATIENCE = 3
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR   = 0.3
MONITOR_METRIC  = "val_f1"  # "val_f1" (Macro-F1, recommended for imbalanced ECG) or "val_loss"

BEST_MODEL_PATH = SAVED_MODELS_DIR / "best_ecg_cnn_bilstm.pt"
