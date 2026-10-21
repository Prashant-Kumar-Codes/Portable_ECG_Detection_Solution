# AI-Based Single-Lead ECG Arrhythmia Detection
**1D CNN-BiLSTM | VitalDB Dataset | PyTorch**

## Project Summary
End-to-end ML pipeline for classifying single-lead ECG recordings (Lead II) into 5 arrhythmia categories using a hybrid 1D CNN + Bidirectional LSTM architecture trained on the VitalDB Arrhythmia Database (482 surgical patients).

## 5 Output Classes
| # | Label | Description |
|---|-------|-------------|
| 0 | Normal Sinus Rhythm | Clean NSR signal |
| 1 | Atrial Fibrillation | Irregular R-R, absent P-waves |
| 2 | Supraventricular Arrhythmia | SVT, PAE, WAP/MAT, SND, AVB |
| 3 | Ventricular Arrhythmia | VT, PVE |
| 4 | Noise / Artifact | Noisy or poor-quality signal |

## Repository Structure
Modules/config.py -- All constants (single source of truth)
Modules/preprocessing.py -- Shared DSP pipeline
Modules/dataset_builder.py -- VitalDB downloader, windowing, ECGWindowDataset
Modules/model.py -- ECGCNNBiLSTM PyTorch architecture
Modules/train.py -- Trainer, evaluate(), class weights
Notebooks/N1_Data_Understanding.ipynb -- Dataset exploration
Notebooks/N2_Preprocessing.ipynb -- DSP pipeline validation
Notebooks/N3_Dataset_Builder.ipynb -- Build windowed .npz cache
Notebooks/N4_Model.ipynb -- Architecture inspection
Notebooks/N5_Training.ipynb -- Training run and evaluation
Dataset/ecg_1_lead_dataset/ -- Raw annotation CSVs and metadata
Dataset/Output_Data/ -- Generated .npz files and saved models

## Quickstart
pip install torch numpy scipy pandas scikit-learn matplotlib seaborn vitaldb

Run notebooks in order: N1 -> N2 -> N3 (builds dataset, 20-60 min) -> N4 -> N5

## Model Architecture
Input: (B, 1, 2500)  [batch, channel, 10s at 250Hz]
1D CNN (4 blocks): 2500 -> 78 tokens, 128 channels
BiLSTM (2 layers, hidden=128, bidirectional): 78 tokens -> (B, 256)
Classifier: Linear(256, 64) -> ReLU -> Dropout -> Linear(64, 5)
Output: (B, 5) raw logits
Total parameters: 782,213

## Preprocessing Pipeline
1. Replace NaN/Inf with 0.0
2. Resample 500 Hz -> 250 Hz (scipy.resample_poly, 1:2 ratio)
3. Zero-phase Butterworth bandpass filter (0.5-40 Hz, order=4, filtfilt)
4. Per-window Z-score normalisation



## Install dependencies (one-time)
``` run commands
pip install -r requirements.txt
# --- Mock mode (no hardware, uses synthetic waveform) ---
python run_live.py
# --- With ESP32 on Windows ---
python run_live.py --port COM3
# --- With ESP32 on Linux ---
python run_live.py --port /dev/ttyUSB0
# --- GPU inference ---
python run_live.py --port COM3 --device cuda
```