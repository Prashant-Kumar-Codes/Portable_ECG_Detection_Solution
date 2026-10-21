# Tasks - Sprint Tracker

## Phase 1: Data Understanding (N1_Data_Understanding.ipynb) [DONE]
- [x] Load metadata.csv (482 patients)
- [x] Explore annotation CSV schema (time_second, beat_type, rhythm_label, bad_signal_quality)
- [x] Verify VitalDB API access
- [x] Fix vdb.load_case() - returns (N,1) array; correct index is vals[:,0]

## Phase 2: Preprocessing (N2_Preprocessing.ipynb + Modules/preprocessing.py) [DONE]
- [x] Implement preprocess_ecg_window() with resample (500->250Hz), Butterworth bandpass (0.5-40Hz), z-score
- [x] Validate pipeline on sample cases with step-by-step visual plots
- [x] Unit test edge cases (NaN handling, constant signal, short signal validation)

## Phase 3: Dataset Builder (N3_Dataset_Builder.ipynb + Modules/dataset_builder.py) [DONE]
- [x] Build class label mapper (supports both full strings and VitalDB abbreviations: N, AFIB/AFL, SVTA, VT, WAP/MAT, SND, AVB)
- [x] Majority-vote window labeler with bad_signal_quality / noise override (>30% threshold)
- [x] Build sliding window extraction loop (stride=500 samples, skip windows with >50% NaN)
- [x] Patient-wise train/val/test split (80/10/10) to prevent subject data leakage
- [x] Safe caching to Dataset/Output_Data/*.npz using Path.exists() guards (skip if exists)
- [x] Log class distribution per split

## Phase 4: Model (N4_Model.ipynb + Modules/model.py) [DONE]
- [x] Implement ConvBlock (1D Conv + BatchNorm + LeakyReLU + MaxPool + Dropout)
- [x] Implement ECGCNNBiLSTM (4 CNN blocks, 2-layer BiLSTM, Classifier Head)
- [x] Verify forward pass tensor shapes through each stage (output shape: (B, 5))
- [x] Verified parameter count: 782,213 parameters

## Phase 5: Training & Persistence (N5_Training.ipynb + Modules/train.py) [DONE]
- [x] Compute robust balanced class weights for CrossEntropyLoss (handles absent classes with neutral 1.0 weight)
- [x] Implement training loop with ReduceLROnPlateau and EarlyStopping
- [x] Checkpoint persistence guarded with Path.exists() (skips retraining if best checkpoint exists)
- [x] Generate training history plots (loss, accuracy, macro-F1)
- [x] Test evaluation with classification report and confusion matrix plotting

## Phase 6: Real-Time Hardware Inference Server [DONE]
- [x] Modules/inference.py — InferenceEngine: loads .pt checkpoint, preprocess + forward pass, returns class/confidence/probs
- [x] Modules/inference.py — Lead-off detection awareness: auto-routes to Noise / Artifact with lead_off flag
- [x] Modules/serial_reader.py — SerialReader: thread-safe 2500-sample ring buffer; real ESP32 serial + mock synthetic-ECG fallback
- [x] Modules/serial_reader.py — Multi-format parser: supports CSV ("1534,0"), JSON ({"v":...,"lo":...}), and plain int values
- [x] Modules/serial_reader.py — Health telemetry: tracks hardware connection state, sample rate (Hz), and samples received
- [x] Modules/server.py — FastAPI: GET / serves dashboard, WS /ws/ecg streams ECG + predictions every 2 sec + status heartbeat
- [x] static/index.html — Dashboard: dynamic auto-scaling Chart.js ECG waveform + real-time health diagnostic pills
- [x] static/index.html — 10-Second Warmup Tracker: live progress bar (0-10s) with calibration state transition
- [x] static/index.html — Signal metrics: Live ADC, Peak-to-Peak (Vpp), baseline, and estimated BPM ticker
- [x] run_live.py — Single entry point: CLI args, auto port collision resolution (WinError 10048 guard), auto-opens browser
- [x] firmware/esp32_ad8232_250hz.ino — 250 Hz hardware timer ISR, streams sample frames @ 115200 baud (fixed syntax typo)
- [x] requirements.txt — All runtime dependencies (torch, numpy, scipy, fastapi, uvicorn, pyserial)

## Phase 7: Next Steps (TODO after the hardware part work well)
- [ ] PostgreSQL session/prediction logging (SQLAlchemy models)
- [ ] PDF report generation (ReportLab)
- [ ] Docker-compose for backend + database deployment
