# Session Memory and Decisions

## Critical Bug Found (N1 Notebook)
vitaldb.load_case(case_id, ['SNUADC/ECG_II'], 1/500) returns shape (N, 1).
Correct indexing: ecg_data = vals[:, 0]  NOT vals[:, 1] (IndexError with size 1)

## VitalDB API Notes
- vdb.load_case(case_id, ['SNUADC/ECG_II'], 1/500) -> numpy array (N,1) at 500Hz
- Some case_ids may have no ECG_II track -> check for all-NaN arrays before processing
- Add 0.5s sleep between downloads to avoid rate limiting
- Annotations are time_second (R-peak timestamps in seconds), NOT sample indices

## Rhythm Label Abbreviations in metadata.csv rhythm_classes column
N=Normal Sinus Rhythm, AFIB/AFL=Atrial Fibrillation, SVTA=Supraventricular Tachyarrhythmia
VT=Ventricular Tachyarrhythmia, SND=Sinus Node Dysfunction, WAP/MAT=Wandering Atrial Pacemaker

## Annotation rhythm_label exact strings (from Annotation_Files CSVs)
VitalDB annotation CSV files use short abbreviations for beats:
- Class 0 (Normal Sinus Rhythm): 'N', 'Normal Sinus Rhythm'
- Class 1 (Atrial Fibrillation): 'AFIB/AFL', 'Atrial Fibrillation', 'Atrial fibrillation'
- Class 2 (Supraventricular Arrhythmia): 'SVTA', 'Patterned Atrial Ectopy', 'WAP/MAT', 'SND', 'AVB'
- Class 3 (Ventricular Arrhythmia): 'VT', 'Patterned Ventricular Ectopy'
- Class 4 (Noise / Artifact): 'Noise', 'Unclassifiable', or when bad_signal_quality == True

All mapped in config.RHYTHM_TO_CLASS.

## Class Weight Computation Fix
- sklearn.utils.class_weight.compute_class_weight raises ValueError if any class in `classes` has 0 occurrences in `y`.
- train.compute_class_weights() computes inverse-frequency weights using NumPy directly:
  weight_k = n_samples / (n_active_classes * count_k)
  For classes with count_k == 0, weight is set to 1.0 (neutral) to prevent division by zero or crashes.

## Safe Data Persistence Decision
- Always check Path.exists() before saving .npz arrays, DataFrames, or model checkpoints.
- Dataset builder checks target split files before processing/downloading VitalDB cases to conserve network and disk resources.
- Trainer checks best model checkpoint before running training epochs.
- Use force_rebuild=True or force_retrain=True to override when needed.

## Preprocessing Decisions
- resample_poly preferred over scipy.signal.resample (integer ratio 2:1, no frequency leakage)
- filtfilt (zero-phase) required for training+inference consistency
- Z-score per window, not per patient or global

## Dataset Split Decision
## Normal Sinus Rhythm & Atrial Fibrillation Fix Verification (2026-10-21)
- Root Cause: VitalDB annotation CSVs use beat abbreviations ('N', 'AFIB/AFL', etc.). The initial RHYTHM_TO_CLASS dictionary had only long strings, causing all normal and atrial fibrillation beats to be mapped to None and skipped.
- Fix: config.RHYTHM_TO_CLASS updated with all shorthand strings ('N', 'AFIB/AFL', 'SVTA', 'VT', etc.).
- Real-data test confirmed: Case 337 produced hundreds of Class 0 (Normal) windows; Case 1023 produced 628 Class 1 (AFIB) windows.
- To produce non-zero support in the final trained model, N3_Dataset_Builder.ipynb Cell 5 must be run to completion (FORCE_REBUILD=True) to generate the new .npz cache files, followed by retraining in N5_Training.ipynb with force_retrain=True.

## Hardware & Serial Acquisition Decisions (2026-10-22)
- **ESP32 ADC in ISR Panic**: `analogRead()` invokes FreeRTOS mutexes and ADC calibration that are not IRAM-safe. Calling it inside a timer interrupt (`onTimer()`) triggers an interrupt watchdog timeout crash (`Guru Meditation Error: Core 1 panic'ed`). Non-blocking `micros()` timing inside `loop()` is universally stable across all ESP32 and Arduino cores.
- **PySerial DTR/RTS Reset Line Lock**: Windows serial drivers assert DTR and RTS by default when opening a COM port. On ESP32 boards with auto-reset circuitry (CH340/CP2102), this holds the `EN` pin LOW in hardware reset. Setting `ser.dtr = False; ser.rts = False` immediately after opening prevents board freezing.
- **Bluetooth Virtual COM Trap**: Windows assigns Bluetooth serial links (e.g., `COM3`, `COM4`) that open successfully in PySerial but never produce bytes. Port auto-detection explicitly filters for physical USB-UART bridges (`CH340`, `CP210`, `FTDI`).
- **USB Cable PnP Code 45**: Charge-only USB cables power on the ESP32 and AD8232 (LEDs light up) but lack D+/D- data lines. Device Manager shows `Problem 45: Device not connected`. A full 4-wire data cable is required.
- **Dynamic Liveness Tracking**: `SerialReader` tracks incoming byte timestamps. If no packets arrive for 3 seconds, `hardware_connected` drops to `False` and sample rate drops to 0.0 Hz, triggering diagnostic hints in the UI.

## Model Training & Metric Monitoring Decisions (2026-10-22)
- **Monitoring Metric (Loss vs. Macro-F1)**: Cross-entropy loss on imbalanced ECG datasets can rise as the network becomes confident/penalized on minority classes even while discrete accuracy and Macro-F1 continue to improve. `MONITOR_METRIC` is configurable in `config.py` (default: `"val_f1"`, `mode="max"`), capturing the best clinical diagnostic checkpoint.
- **Side-by-Side Train vs. Val Metrics**: `train.py` computes both training and validation metrics (`loss`, `accuracy`, `macro_f1`) per epoch, allowing detection of overfitting or underfitting. Checkpoints store `train_metrics`, `val_metrics`, and the full 6-metric `history`.
- **Jupyter Notebook Caching**: `import config` is cached in `sys.modules`. Edits to `config.py` on disk do not propagate to a running notebook without a kernel restart or `%load_ext autoreload; %autoreload 2`. Explicit keyword arguments in `Trainer(max_epochs=..., patience=..., monitor=...)` should be supported for interactive use.

