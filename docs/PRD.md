Here is your updated, formal **Project Description** reflecting the switch to the **VitalDB Arrhythmia Database** and the **1D CNN-BiLSTM Hybrid Architecture**, followed by a compact **Project Memory (Master Context Block)** that you can save, pin, or paste into future sessions so we never lose track of your exact technical constraints.

---

# Part 1: Updated Project Description

## AI-Based Single-Lead ECG Arrhythmia Detection Using Hardware and Hybrid Deep Learning (1D CNN-BiLSTM)

### 1. Project Overview

The **AI-Based Single-Lead ECG Arrhythmia Detection System** is a hardware-integrated, full-stack web application designed to acquire single-lead electrocardiogram (ECG) signals from a human subject using a three-electrode ECG sensor and analyze them in real time using a hybrid deep learning architecture (**1D CNN-BiLSTM**).

Unlike traditional models trained on 12-lead clinical datasets that suffer from domain shift when deployed on low-cost single-lead hardware, this system trains natively on single-lead (`ECG_II`) recordings from the **VitalDB Arrhythmia Database**. By combining 1D Convolutional layers (for morphological P-QRS-T feature extraction) with Bidirectional Long Short-Term Memory networks (for temporal R-R interval rhythm modeling), the system identifies cardiac rhythm abnormalities and signal noise, streaming real-time waveforms and diagnostic insights to a web dashboard.

### 2. Project Objectives

* **Real-Time Hardware Acquisition:** Acquire single-channel (Lead II configuration) analog ECG signals at a deterministic sampling rate of **250 Hz** using an **AD8232** sensor module and **ESP32** microcontroller.
* **Wired Serial Streaming:** Transmit sequential, timestamped ADC samples to a host computer over USB Serial without wireless packet drop.
* **Unified Signal Preprocessing:** Apply an identical digital signal processing (DSP) pipeline—comprising resampling, a 4th-order Butterworth bandpass filter (0.5–40 Hz), and Z-score window normalization—across both training data and live hardware inference.
* **Hybrid Deep Learning Inference:** Train and deploy a PyTorch **1D CNN-BiLSTM** model using the **VitalDB Arrhythmia Database** to classify 10-second ECG windows into 5 actionable categories, including a dedicated hardware noise/artifact class.
* **Interactive Live Dashboard:** Stream real-time ECG waveforms, signal quality indicators, and model confidence scores to a **React + Plotly** frontend via **FastAPI WebSockets**.
* **Persistence & Reporting:** Log session metadata and arrhythmia events in a **PostgreSQL** database and generate downloadable clinical-style PDF summary reports.

### 3. Hardware Components

* **ESP32 Development Board:** Samples the analog output of the AD8232 via its 12-bit ADC using a hardware timer interrupt locked at **250 Hz** (4 ms intervals) and streams packets over USB UART.
* **AD8232 ECG Sensor Module:** Single-lead analog front-end (AFE) IC that amplifies bio-potentials, rejects common-mode interference, and provides hardware lead-off detection (`LO+` and `LO-` pins).
* **Three-Lead ECG Electrodes:** Ag/AgCl gel electrodes placed in a standard Lead II triangle (RA, LA, RL) to match the `SNUADC/ECG_II` lead configuration of the VitalDB dataset.
* **USB Data Cable & Electrical Isolation:** Connects the ESP32 to the host computer (operated on laptop battery power or via a USB isolator during human testing to prevent 50/60 Hz mains ground loops and ensure safety).

### 4. Updated Software and Technology Stack

| Component | Technology | Role & Specification |
| --- | --- | --- |
| **Microcontroller Firmware** | C/C++ (Arduino / ESP-IDF) | Hardware timer-driven 250 Hz ADC sampling + Lead-Off pin status over UART (`115200` baud) |
| **Serial Acquisition** | Python, `PySerial`, `asyncio` | Non-blocking ring buffer reading incoming serial frames |
| **Signal Processing (DSP)** | `NumPy`, `SciPy` | 500 Hz $\rightarrow$ 250 Hz resampling, 0.5–40 Hz Butterworth bandpass filter, Z-score normalization |
| **Dataset** | **VitalDB Arrhythmia Database (v1.0.0)** | Single-lead (`SNUADC/ECG_II`), 482 patients, fetched via `vitaldb` Python API + PhysioNet CSV annotations |
| **Deep Learning Architecture** | **PyTorch (1D CNN-BiLSTM)** | 4-block 1D CNN feature extractor ($2500 \rightarrow 78$ steps) + 2-layer Bidirectional LSTM + Linear Classifier |
| **Backend API** | Python, **FastAPI**, Pydantic, WebSockets | Real-time waveform broadcasting, inference orchestration, and REST endpoints |
| **Frontend Dashboard** | **React** (TypeScript), **Plotly.js**, Tailwind CSS | Live scrolling ECG monitor, confidence gauges, and historical playback |
| **Database** | **PostgreSQL** + SQLAlchemy | Stores recording sessions, abnormality flags, confidence scores, and timestamps |
| **Report Generation** | Python (`ReportLab` / `WeasyPrint`) | Generates downloadable PDF reports with representative ECG strips and summary statistics |
| **Deployment** | Docker, Nginx | Containerized backend, database, and frontend services |

### 5. Working Methodology

1. **Step 1: Native Single-Lead Dataset Preparation:** Using the PhysioNet VitalDB Arrhythmia annotation CSVs and the `vitaldb` Python library, raw 500 Hz `ECG_II` waveforms are fetched, downsampled to **250 Hz**, segmented into **10-second windows (2,500 samples)**, and split strictly by `case_id` (patient-wise split) to prevent data leakage.
2. **Step 2: Signal Preprocessing:** Both dataset windows and live ESP32 windows pass through the exact same `preprocess_ecg_window()` pipeline: NaN removal, 4th-order zero-phase Butterworth bandpass filtering (**0.5 Hz to 40 Hz**) to eliminate baseline wander and powerline hum, and **per-window Z-score normalization** ($\mu=0, \sigma=1$) to make the model invariant to voltage scaling differences between clinical monitors (mV) and the AD8232 ADC (0–4095).
3. **Step 3: Hybrid 1D CNN-BiLSTM Model Analysis:**
* **CNN Front-End:** Extracts local morphological waveforms (P-wave, QRS complex width, T-wave inversion) and compresses the sequence length from 2,500 samples down to 78 temporal feature tokens.
* **BiLSTM Back-End:** Analyzes sequential dependencies across the 78 tokens in both forward and backward directions to capture R-R interval irregularities over the 10-second window.
* **Classification:** Outputs probabilities across 5 target classes: **Normal Sinus Rhythm**, **Atrial Fibrillation**, **Supraventricular Arrhythmia**, **Ventricular Arrhythmia**, and **Noise / Poor Signal Quality**.


4. **Step 4: Real-Time Acquisition & Sliding Window Inference:** The ESP32 streams 250 samples/sec over USB. The Python backend maintains a 10-second circular buffer (2,500 samples), streaming raw/filtered chunks at 20 FPS to the React frontend via WebSockets while triggering PyTorch model inference every 2 to 5 seconds.
5. **Step 5: Visualization, Persistence & Reporting:** The React dashboard plots the live trace and displays the predicted rhythm class alongside model confidence. Completed sessions and detected arrhythmia episodes are committed to PostgreSQL and can be exported as a formatted PDF report.

### 6. System Architecture Flow

```text
[3-Lead Electrodes (Lead II)] 
       │
       ▼
[AD8232 Analog Front-End] 
       │ (Analog Voltage + LO+/LO- Status)
       ▼
[ESP32 Microcontroller] ──(250 Hz Hardware Timer Interrupt)
       │
       ▼ (USB Serial UART @ 115200 Baud)
[Python PySerial Ring Buffer (2500-Sample Sliding Window)]
       │
       ▼
[SciPy DSP Pipeline: 0.5-40Hz Bandpass + Z-Score Normalization]
       ├──► (FastAPI WebSocket Stream) ──────► [React + Plotly Real-Time Waveform UI]
       ▼                                                    ▲
[PyTorch 1D CNN-BiLSTM Inference Engine] ───────────────────┘ (Class + Confidence)
       │
       ▼
[PostgreSQL History DB] ──► [Python PDF Report Generator]

```

### 7. Scope and Limitations

* **Single-Lead Constraint:** Because a 3-electrode Lead II setup captures cardiac electrical activity along a single vector, the system focuses on **rhythm abnormalities** and **ectopic beat patterns** detectable in Lead II rather than localized myocardial ischemia/infarction (which require 12-lead spatial views).
* **Perioperative Dataset Context:** The VitalDB Arrhythmia Database is collected from surgical patients; while anesthesiologist-validated with high agreement ($\kappa = 0.93$), resting ambulatory signals may exhibit different baseline heart rate distributions.
* **Educational & Research Prototype:** Predictions are strictly for educational and research screening and must not be used for clinical diagnosis.

---

# Part 2: Project Memory (Master Context Specification)

> **How to use this section:** Copy this block into a local `PROJECT_MEMORY.md` file in your code repository, or paste it at the start of any new chat session so the AI immediately knows your exact tensor shapes, sampling rates, label mappings, and architecture decisions.

```markdown
# PROJECT MEMORY: AI-ECG Arrhythmia Detection System

## 1. Core Technical Decisions & Locked Constants
- **Lead Configuration:** Single-Lead (Lead II / `SNUADC/ECG_II`).
- **Dataset:** VitalDB Arrhythmia Database v1.0.0 (PhysioNet + `vitaldb` Python package).
- **Original Dataset Sampling Rate:** 500 Hz.
- **Target System Sampling Rate (`TARGET_FS`):** 250 Hz (4 ms per sample on ESP32; VitalDB downsampled 2:1).
- **Window Duration (`WINDOW_SEC`):** 10 seconds.
- **Model Input Tensor Shape:** `(Batch_Size, 1, 2500)` -> `[Channels=1, Sequence_Length=2500]`, dtype `float32`.
- **Sliding Window Stride (Live Inference):** 2.0 seconds (500 new samples per inference trigger).

## 2. Shared Preprocessing Pipeline (`preprocess_ecg_window`)
MUST be identical in training and FastAPI live inference:
1. Replace `NaN` / `Inf` with `0.0`.
2. Resample to `250 Hz` (if input is 500 Hz VitalDB data).
3. Filter: 4th-order zero-phase Butterworth Bandpass (`0.5 Hz` to `40.0 Hz` via `scipy.signal.butter` + `filtfilt`).
4. Normalize: Per-window Z-score normalization `x_norm = (x - mean(x)) / (std(x) + 1e-6)`.
   *(Critical: Bridges the scale gap between VitalDB mV floats and ESP32 12-bit ADC integers 0-4095).*

## 3. Target Classes (`NUM_CLASSES = 5`)
Mapped from VitalDB `rhythm_label` and `bad_signal_quality`:
- `0`: **Normal Sinus Rhythm** (`Normal Sinus Rhythm` & `bad_signal_quality == False`)
- `1`: **Atrial Fibrillation** (`Atrial Fibrillation` & `bad_signal_quality == False`)
- `2`: **Supraventricular Arrhythmia** (`Supraventricular Tachyarrhythmia`, `Patterned Atrial Ectopy`)
- `3`: **Ventricular Arrhythmia** (`Ventricular Tachyarrhythmia`, `Patterned Ventricular Ectopy`)
- `4`: **Noise / Artifact** (`Noise` OR `bad_signal_quality == True` OR ESP32 `LO+/LO-` lead-off triggered)

## 4. Deep Learning Model Specification (`ECG_CNN_BiLSTM`)
- **Framework:** PyTorch
- **Stage 1 (1D CNN Feature Extractor):**
  - ConvBlock 1: `Conv1d(1, 32, k=15, s=2, p=7)` + `BatchNorm1d` + `ReLU` + `MaxPool1d(2, 2)` -> `L=625`
  - ConvBlock 2: `Conv1d(32, 64, k=7, s=1, p=3)` + `BatchNorm1d` + `ReLU` + `MaxPool1d(2, 2)` -> `L=312`
  - ConvBlock 3: `Conv1d(64, 128, k=5, s=1, p=2)` + `BatchNorm1d` + `ReLU` + `MaxPool1d(2, 2)` -> `L=156`
  - ConvBlock 4: `Conv1d(128, 128, k=3, s=1, p=1)` + `BatchNorm1d` + `ReLU` + `MaxPool1d(2, 2)` -> `L=78`
- **Stage 2 (Temporal Sequence Modeling):**
  - Permute `(B, 128, 78)` -> `(B, 78, 128)`
  - `nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)`
  - Global Average Pooling over time dimension (`dim=1`) -> `(B, 256)`
- **Stage 3 (Classifier Head):**
  - `Linear(256, 64)` -> `ReLU` -> `Dropout(0.3)` -> `Linear(64, 5)`
- **Validation Strategy:** Strict patient-wise split by `case_id` (80% Train, 10% Val, 10% Test) with class-weighted `CrossEntropyLoss`.

## 5. Hardware & Full-Stack Integration Contract
- **ESP32 Pinout:** `AD8232 OUTPUT -> GPIO 34 (ADC1_CH6)`, `LO+ -> GPIO 32`, `LO- -> GPIO 33`, `3.3V`, `GND`.
- **Serial Protocol:** `115200` baud, newline-delimited ASCII or compact JSON frames at `250 Hz`.
- **Backend:** FastAPI + `PySerial` background task + WebSocket `/ws/ecg` + PostgreSQL (`sessions`, `predictions` tables).
- **Frontend:** React + TypeScript + Plotly.js (renders 10-second rolling buffer) + PDF export endpoint (`/api/reports/{session_id}/pdf`).

```

---

## Suggested Repository Folder Structure

To keep your training scripts, hardware firmware, backend, and React app cleanly separated as we build this out, set up your project directory like this:

```text
ai-ecg-arrhythmia-system/
├── PROJECT_MEMORY.md           # The master context file above
├── firmware/
│   └── esp32_ad8232_250hz/
│       └── esp32_ad8232_250hz.ino
├── ml_pipeline/
│   ├── data/                   # Downloaded PhysioNet Annotation_Files/ & metadata.csv
│   ├── processed_cache/        # Cached .npz patient windows so you only download once
│   ├── dataset_builder.py      # VitalDB downloader, 0.5-40Hz filter, patient-wise splitter
│   ├── model.py                # ECG_CNN_BiLSTM PyTorch architecture
│   ├── train.py                # Training loop, class weights, F1/Confusion Matrix logging
│   └── saved_models/
│       └── best_ecg_cnn_bilstm.pt
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI entry point & WebSocket manager
│   │   ├── serial_reader.py    # PySerial 250Hz async consumer
│   │   ├── inference.py        # Imports preprocess_ecg_window & loads .pt weights
│   │   ├── database.py         # PostgreSQL SQLAlchemy models
│   │   └── pdf_generator.py    # ReportLab PDF builder
│   └── requirements.txt
└── frontend/                   # React + TypeScript + Plotly dashboard

```