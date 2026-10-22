# System Architecture - AI-Based Single-Lead ECG Arrhythmia Detection

## Data Flow
- **Offline Training Pipeline**:
  VitalDB API (500 Hz) -> preprocessing.py (resample+filter+normalize) -> dataset_builder.py (2500-sample windows, class labels, .npz cache) -> model.py (ECGCNNBiLSTM) -> train.py
- **Live Hardware Inference Pipeline**:
  AD8232 ECG Sensor -> ESP32 / Arduino (250 Hz UART) -> `serial_reader.py` (Multi-format CSV/JSON parser + Ring Buffer) -> `server.py` (FastAPI + WebSocket `/ws/ecg`) -> `inference.py` (ECGCNNBiLSTM 2s stride) -> `static/index.html` (Chart.js live dashboard)

## Module Dependency Graph
- `config.py`: constants (paths, FS, classes, hyper-params: `MONITOR_METRIC="val_f1"`, `EARLY_STOP_PATIENCE=3`, `MAX_EPOCHS=50`)
- `preprocessing.py`: `preprocess_ecg_window()` (shared DSP filter + Z-score)
- `dataset_builder.py`: `VitalDBDatasetBuilder`, `ECGWindowDataset`
- `model.py`: `ECGCNNBiLSTM`, `ConvBlock`
- `train.py`: `Trainer` (computes train and val loss/accuracy/macro-F1 per epoch, early stopping on configured metric, checkpoints with 6-metric history), `evaluate()`, `compute_class_weights()`
- `serial_reader.py`: `SerialReader` (thread-safe 2500-sample ring buffer, DTR/RTS reset isolation, auto-reconnect, dynamic liveness tracker, multi-format CSV/JSON/3-token parser)
- `inference.py`: `InferenceEngine` (model loader, lead-off handling, softmax probabilities)
- `server.py`: FastAPI application serving dashboard and streaming WebSocket telemetry
- `run_live.py`: CLI entry point with USB-UART bridge auto-detection, port collision resolver, and browser launcher
- `firmware/esp32_ad8232_250hz/esp32_ad8232_250hz.ino`: Non-blocking 250 Hz `micros()` loop with GPIO 2 heartbeat LED

## Key Constants
TARGET_FS=250, SOURCE_FS=500, WINDOW_SEC=10, WINDOW_SAMPLES=2500, STRIDE_SAMPLES=500, NUM_CLASSES=5, BATCH_SIZE=64, LR=1e-3, MAX_EPOCHS=50, EARLY_STOP_PATIENCE=3, MONITOR_METRIC="val_f1"

## CNN Architecture (input: B x 1 x 2500)
CB1: Conv1d(1,32,k=15,s=2,p=7) + BN + ReLU + MaxPool1d(2) -> L=625
CB2: Conv1d(32,64,k=7,p=3) + BN + ReLU + MaxPool1d(2) -> L=312
CB3: Conv1d(64,128,k=5,p=2) + BN + ReLU + MaxPool1d(2) -> L=156
CB4: Conv1d(128,128,k=3,p=1) + BN + ReLU + MaxPool1d(2) -> L=78
BiLSTM(128->256 bidirectional, 2 layers) -> GlobalAvgPool -> Linear(256,64) -> Linear(64,5)

## Class Mapping
0=Normal Sinus Rhythm, 1=Atrial Fibrillation, 2=Supraventricular Arrhythmia, 3=Ventricular Arrhythmia, 4=Noise / Artifact

