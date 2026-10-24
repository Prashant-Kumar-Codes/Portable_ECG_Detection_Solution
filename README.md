# AI-Based Single-Lead ECG Arrhythmia Detection
**1D CNN-BiLSTM | VitalDB Dataset | Real-Time Hardware Inference | PyTorch**

---

## 1. Project Summary
End-to-end deep learning system for classifying single-lead ECG recordings (Lead II) into 5 diagnostic arrhythmia categories using a hybrid **1D CNN + Bidirectional LSTM** architecture.

- **Offline Training**: Trained on the VitalDB Arrhythmia Database (482 surgical patients, 500 Hz resampled to 250 Hz).
- **Online Telemetry**: Connects directly to an AD8232 ECG sensor via ESP32 / Arduino serial UART (250 Hz) or mock generator.
- **Live AI Dashboard**: Real-time browser dashboard featuring dynamic auto-scaling waveforms, 10-second calibration/recording tracking, system health diagnostics, and continuous classification every 2 seconds.

---

## 2. 5 Output Classes
| # | Label | Clinical Description |
|---|-------|----------------------|
| **0** | **Normal Sinus Rhythm** | Regular rhythm, consistent P-QRS-T morphology |
| **1** | **Atrial Fibrillation** | Irregular R-R intervals, absence of distinct P-waves |
| **2** | **Supraventricular Arrhythmia** | Atrial premature beats, SVT, AV blocks, junctional rhythms |
| **3** | **Ventricular Arrhythmia** | Ventricular tachycardia, premature ventricular contractions |
| **4** | **Noise / Artifact** | Baseline wander, electrode disconnect, motion artifacts |

---

## 3. Hardware Architecture & Pinout
Connect the **AD8232 Heart Rate Monitor Module** to your ESP32 or Arduino as follows:

| AD8232 Pin | ESP32 Pin | Arduino Uno/Nano | Description |
|------------|-----------|------------------|-------------|
| **OUTPUT** | GPIO 34 (ADC1_CH6) | A0 | Analog ECG signal (0–3.3V / 0–5V) |
| **LO+**    | GPIO 32   | D10 | Lead-Off positive detection |
| **LO-**    | GPIO 33   | D11 | Lead-Off negative detection |
| **3.3V**   | 3.3V      | 3.3V / 5V | Sensor power supply |
| **GND**    | GND       | GND | Common ground |

Electrode placement (**Lead II standard**):
- **RA (Red)**: Right clavicle / right arm.
- **LA (Yellow)**: Left clavicle / left arm.
- **RL (Green)**: Right lower abdomen / right leg (reference ground).

---

## 4. Supported Serial Protocols (115200 Baud)
The Python acquisition engine automatically detects and parses any of the following newline-delimited formats:

1. **Comma-Separated Values (CSV)** *(Standard Arduino/ESP32)*:
   ```text
   1534,0
   1607,0
   ```
   *(Format: `<ADC_Value>,<Lead_Off_Flag>` or `<ADC_Value>,<LO+>,<LO->`)*

2. **JSON Stream** *(From `firmware/esp32_ad8232_250hz/esp32_ad8232_250hz.ino`)*:
   ```json
   {"v": 2048, "lo": 0}
   ```
   Signal fields `v`, `ecg`, `val`, `value`, `adc`, `raw`, and `signal` are supported (case-insensitively). Lead-off fields `lo`, `lead_off`, `leads_off`, and `leadoff` accept boolean, numeric, or common string values.
   JSON objects without a recognized signal field are skipped instead of being treated as zero-valued ECG samples.

3. **Plain Raw ADC Integers**:
   ```text
   1534
   ```

---

## 5. DSP Preprocessing Pipeline
Applied identically during model training and real-time live inference:
1. **Sanitisation**: Replace `NaN` and `Inf` with `0.0`.
2. **Resampling**: 500 Hz downsampled to 250 Hz (`scipy.signal.resample_poly`, 1:2 ratio). Live hardware streams natively at 250 Hz.
3. **Bandpass Filtering**: 4th-order zero-phase Butterworth filter (0.5–40 Hz) via `scipy.signal.filtfilt` to remove baseline drift and high-frequency powerline noise.
4. **Per-Window Z-Score Normalisation**: $\frac{x - \mu}{\sigma + 10^{-6}}$. Normalises amplitude differences between patient skin impedance and ADC resolutions.

---

## 6. Model Architecture & Training
- **Input**: $(B, 1, 2500)$ — 10 seconds of Lead-II ECG @ 250 Hz.
- **1D CNN Feature Extractor (4 Blocks)**:
  - Block 1: `Conv1d(1, 32, k=15, s=2, p=7)` + BatchNorm + ReLU + MaxPool(2) $\rightarrow L=625$
  - Block 2: `Conv1d(32, 64, k=7, p=3)` + BatchNorm + ReLU + MaxPool(2) $\rightarrow L=312$
  - Block 3: `Conv1d(64, 128, k=5, p=2)` + BatchNorm + ReLU + MaxPool(2) $\rightarrow L=156$
  - Block 4: `Conv1d(128, 128, k=3, p=1)` + BatchNorm + ReLU + MaxPool(2) $\rightarrow L=78$
- **Temporal Sequence Modeling**: 2-layer Bidirectional LSTM (hidden dimension 128) with Global Average Pooling $\rightarrow (B, 256)$.
- **Classification Head**: `Linear(256, 64)` $\rightarrow$ ReLU $\rightarrow$ Dropout(0.3) $\rightarrow$ `Linear(64, 5)`.
- **Total Parameters**: 782,213.
- **Optimization & Metric Monitoring**:
  - Balanced class weights for imbalanced clinical distribution.
  - Checkpointing & Early Stopping default to `MONITOR_METRIC="val_f1"` (`mode="max"`, patience=3) to preserve best discrete classification checkpoints.
  - Per-epoch training & validation metric tracking (`loss`, `accuracy`, `macro_f1`). Checkpoints persist `train_metrics`, `val_metrics`, and complete 6-metric `history`.

---

## 7. Live Telemetry Dashboard Features
- **System Health Badges**:
  - **Hardware Connection**: Displays active port and baud rate (e.g. `COM5 Connected (115200)`) or `Mock Mode`.
  - **Lead-Off Detection**: Instant visual alert (`⚠️ Lead-Off!`) if an electrode detaches from skin.
  - **Real-Time Sampling Rate**: Measures incoming packets per second (target: `250.0 Hz`), dynamically falling to 0.0 Hz if stream halts.
  - **Model Status**: Confirms model weights loaded on CPU/CUDA.
- **Diagnostic Troubleshooting Box**:
  - Automatically pops up actionable troubleshooting hints (cable data support, baud rate, COM selection) if "No Incoming Bytes" is detected.
- **10-Second Warmup & Calibration Tracker**:
  - Displays a live progress bar (`0.0s / 10.0s`) while accumulating the initial 2500-sample clinical diagnostic window.
  - Automatically transitions to continuous inference (2-second sliding stride) once calibrated.
- **Dynamic Waveform Auto-Scaling**:
  - Automatically fits the Y-axis to the sensor's active dynamic range with 20% margin, avoiding flatline compression artifacts.
- **Live Signal Telemetry**:
  - Live ADC count, Peak-to-Peak (Vpp) amplitude, baseline count, and estimated Heart Rate (BPM).

---

## 8. Installation & Running

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Live Server

```bash
# --- 1. Mock Mode (no hardware, synthetic ECG) ---
python run_live.py

# --- 2. Live Hardware (Windows COM Port) ---
python run_live.py --port COM5

# --- 3. Live Hardware (Linux / macOS) ---
python run_live.py --port /dev/ttyUSB0

# --- 4. Custom Port & GPU Acceleration ---
python run_live.py --port COM5 --http-port 8080 --device cuda
```
*Note: If port 8000 is busy, `run_live.py` automatically binds to the next free port and opens the browser.*

### Retrain After Changing Model Sizes

The live server checkpoint must be trained with the current architecture. If CNN/LSTM
sizes have changed, retrain from the cached dataset before starting `run_live.py`:

```powershell
py -c "import sys; sys.path.insert(0, 'Modules'); from train import Trainer; Trainer(skip_if_exists=False).run(force_retrain=True)"
```

This saves new weights to `Dataset/Output_Data/saved_models/best_ecg_cnn_bilstm.pt`.
The training pipeline checks existing checkpoint shapes and will not silently skip
training when the checkpoint was made for an older model architecture.

---

## 9. Repository Structure
```text
├── Dataset/
│   ├── ecg_1_lead_dataset/       # Raw metadata & VitalDB annotation files
│   └── Output_Data/saved_models/ # Saved model checkpoint (best_ecg_cnn_bilstm.pt)
├── Modules/
│   ├── config.py                 # Central configuration constants
│   ├── preprocessing.py          # Shared DSP bandpass + normalisation
│   ├── dataset_builder.py        # Windowing & dataset pipeline
│   ├── model.py                  # PyTorch ECGCNNBiLSTM architecture
│   ├── train.py                  # Training loop & evaluation
│   ├── inference.py              # Real-time inference engine
│   ├── serial_reader.py          # Multi-format serial buffer & metrics
│   └── server.py                 # FastAPI backend & WebSocket stream
├── Notebooks/
│   ├── N1_Data_Understanding.ipynb
│   ├── N2_Preprocessing.ipynb
│   ├── N3_Dataset_Builder.ipynb
│   ├── N4_Model.ipynb
│   └── N5_Training.ipynb
├── docs/                         # Architecture, Design, Tasks, PRD
├── firmware/
│   └── esp32_ad8232_250hz/       # ESP32 250 Hz non-blocking micros() sketch + heartbeat LED
├── static/
│   └── index.html                # Professional live telemetry dashboard
├── run_live.py                   # Unified CLI runner
└── requirements.txt              # Project dependencies
```

---

## 10. Clinical Disclaimer
This system is an **educational and engineering research prototype**. It is not a certified medical device and must not be used as a substitute for professional clinical diagnosis or patient monitoring.