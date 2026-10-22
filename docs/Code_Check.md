# Code Verification & Diagnostics Log

Last Updated: 2026-10-21 (Post-Fix Verification)

---

## 1. Class Support Issue Analysis: Normal Sinus Rhythm & Atrial Fibrillation

### Observed Symptoms
During initial model evaluation in `Notebooks/N5_Training.ipynb`, the classification report showed:
```text
                             precision    recall  f1-score   support

        Normal Sinus Rhythm     0.0000    0.0000    0.0000         0
        Atrial Fibrillation     0.0000    0.0000    0.0000         0
Supraventricular Arrhythmia     0.2321    0.6089    0.3361       202
     Ventricular Arrhythmia     0.8786    0.6464    0.7448       806
           Noise / Artifact     0.9464    0.8987    0.9220      2281

                   accuracy                         0.8191      3289
                  macro avg     0.4114    0.4308    0.4006      3289
               weighted avg     0.8859    0.8191    0.8426      3289
```

### Root Cause
1. **String Discrepancy**: The VitalDB annotation CSV files (`Annotation_file_{case_id}.csv`) annotate beats using clinical shorthand abbreviations:
   - `'N'` for Normal Sinus Rhythm
   - `'AFIB/AFL'` for Atrial Fibrillation / Flutter
   - `'SVTA'`, `'WAP/MAT'`, `'SND'`, `'AVB'` for Supraventricular Arrhythmias
   - `'VT'`, `'Patterned Ventricular Ectopy'` for Ventricular Arrhythmias
   - `'Noise'`, `'Unclassifiable'` for Artifacts
2. The initial `RHYTHM_TO_CLASS` dictionary in [config.py](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/Modules/config.py) only matched long strings like `"Normal Sinus Rhythm"` and `"Atrial Fibrillation"`.
3. Consequently, whenever `'N'` or `'AFIB/AFL'` occurred, `assign_label()` returned `None`, skipping those windows during sliding-window extraction. Only ectopy and noise beats were mapped, resulting in 0 windows for Class 0 and Class 1 in the cached `.npz` files.

---

## 2. Verification of the Code Fix

### A. Configuration & Mapping ([config.py](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/Modules/config.py))
`RHYTHM_TO_CLASS` has been updated to explicitly map all VitalDB rhythm representations:
```python
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
    "Unclassifiable":                                   4,
}
```

### B. Unit Tests in [N3_Dataset_Builder.ipynb](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/Notebooks/N3_Dataset_Builder.ipynb)
- Tested `assign_label('N', False)` -> returns `0` [PASS]
- Tested `assign_label('AFIB/AFL', False)` -> returns `1` [PASS]
- Tested `assign_label('Normal Sinus Rhythm', True)` -> returns `4` (noise override) [PASS]

### C. Live Real-Data Verification
- **Normal Sinus Rhythm**: Verified on dry-run cases (e.g., Case 337: 221 Normal windows; Case 103: 413 Normal windows; Case 244: 518 Normal windows).
- **Atrial Fibrillation**: Verified on real VitalDB data (Case 1023): extracted 647 windows -> 628 Class 1 (AFIB/AFL), 19 Class 4 (Noise).

### E. Dataset Builder & Split Correctness Audit ([dataset_builder.py](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/Modules/dataset_builder.py) & [N3_Dataset_Builder.ipynb](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/Notebooks/N3_Dataset_Builder.ipynb))
- **Patient-wise Split**: 482 unique `case_id`s split 80/10/10 (385 train, 48 val, 49 test) with 0 patient overlap.
- **Rhythm Representation Across Splits**: Verified that all 5 classes are present in every split:
  - `Train (385 cases)`: 292 Normal (N), 93 AFIB/AFL, 89 SVTA, 73 VT, 204 Noise cases.
  - `Val (48 cases)`: 41 Normal (N), 5 AFIB/AFL, 11 SVTA, 8 VT, 24 Noise cases.
  - `Test (49 cases)`: 37 Normal (N), 13 AFIB/AFL, 9 SVTA, 7 VT, 22 Noise cases.
- **Sliding Window & Labeling Logic**:
  - Sample timing: 5000 raw samples at 500 Hz (10s window) stride 1000 samples (2s stride).
  - Resampled cleanly to 2500 samples at 250 Hz (matches ESP32 target fs).
  - R-peak timestamps aligned with `time_second` from VitalDB annotation CSVs.
  - Added `.strip()` to `assign_label` for defensive whitespace handling.
  - Added AFIB Case 1023 to `DRY_RUN_IDS` in N3 Cell 3 so all 5 classes are demonstrated during the pre-build dry run.
- **Split Processing Verification**:
  - Verified sample cases from each split:
    - Train (Case 98): 403 Normal, 148 Ventricular Arrhythmia, 12 Noise windows.
    - Val (Case 4179): 507 Normal, 42 Supraventricular, 26 Noise windows.
    - Test (Case 1733): 270 Normal, 17 AFIB, 4 Supraventricular, 230 Noise windows.
  - Formatted arrays (`X: (N, 2500) float32`, `y: (N,) int64`) are saved via `np.savez_compressed` and loaded by `ECGWindowDataset` as `(N, 1, 2500)`.

---

## 3. Potential Problems & Mitigation Checklist

| Potential Issue | Cause | Mitigation In Place |
| :--- | :--- | :--- |
| **Old Cache Loaded** | Existing `windows_*.npz` files may have been built before the mapping fix | Set `FORCE_REBUILD = True` in [N3_Dataset_Builder.ipynb](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/Notebooks/N3_Dataset_Builder.ipynb) (Cell 5) to download and regenerate the dataset. |
| **Build Interruption** | Downloading 482 cases from VitalDB takes 25–45 minutes over internet; stopping early discards unsaved split | Allow Cell 5 in N3 to run to completion. Status logs report progress every 10 cases. |
| **Model Retrain Skipped** | `Trainer.run()` checks `skip_if_exists=True` and skips training if `best_ecg_cnn_bilstm.pt` is found | Use `trainer.run(force_retrain=True)` in [N5_Training.ipynb](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/Notebooks/N5_Training.ipynb) when retraining on the newly rebuilt dataset. |
| **Loss vs. F1 Checkpoint Miss** | Loss can increase due to probability calibration penalties while Macro-F1 continues to improve | Configured `MONITOR_METRIC = "val_f1"` (`mode="max"`) in [config.py](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/Modules/config.py) and [train.py](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/Modules/train.py) so early stopping and checkpoints capture optimal clinical accuracy. |
| **Jupyter Config Caching** | `import config` cached in `sys.modules`; disk edits to `config.py` not recognized in running kernel | Pass parameters explicitly to `Trainer(...)` or run `%load_ext autoreload; %autoreload 2` in Notebook Cell 1. |
| **ESP32 FreeRTOS ISR Panic** | `analogRead()` called in hardware timer ISR violates IRAM safety, causing `Interrupt wdt timeout` crash loop | Refactored [firmware](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/firmware/esp32_ad8232_250hz/esp32_ad8232_250hz.ino) to non-blocking `micros()` 250 Hz timing in `loop()`; added GPIO 2 heartbeat LED. |
| **PySerial DTR/RTS Reset Lock** | Windows serial driver asserts DTR/RTS on open, locking ESP32 in reset/bootloader mode | [serial_reader.py](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/Modules/serial_reader.py) explicitly clears `ser.dtr = False; ser.rts = False` and resets input buffer. |
| **Virtual Bluetooth Trap** | Windows `COM3`/`COM4` Bluetooth ports open without error but produce 0 bytes | [run_live.py](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/run_live.py) filters out Bluetooth links and prioritizes physical USB-UART bridges (`CH340`, `CP210`, `FTDI`). |
| **USB Cable Missing Data Lines** | Charge-only cable powers ESP32 but lacks D+/D- lines (Windows PnP Code 45) | [index.html](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/static/index.html) auto-displays dynamic diagnostic troubleshooting hints when "No Incoming Bytes" occurs. |
| **Hardware Disconnect** | AD8232 electrode lead detached during real-time streaming | Firmware transmits `lo: 1` (`!LO` packet); [server.py](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/Modules/server.py) flags `lead_off: true`; frontend displays yellow "Lead-Off" overlay. |
| **Missing Serial Hardware** | ESP32 not plugged in when running [run_live.py](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/run_live.py) | [serial_reader.py](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/Modules/serial_reader.py) automatically activates Mock Mode with synthetic 250 Hz ECG. |

---

## 4. Next Step to Finalise Full Model
1. Run **Cell 5** of [N3_Dataset_Builder.ipynb](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/Notebooks/N3_Dataset_Builder.ipynb) until completion to write the new `windows_train.npz`, `windows_val.npz`, and `windows_test.npz` files.
2. Run [N5_Training.ipynb](file:///d:/Codes/Projects/Portable_ECG_Detection_Solution/Notebooks/N5_Training.ipynb) with `force_retrain=True`.
3. Launch the live real-time dashboard with:
   ```bash
   python run_live.py --port COM5
   ```

---

## 5. Live Hardware Acquisition & Training Diagnostic Verification (2026-10-22)

- **Firmware Non-Blocking Sampling**: Verified at 250 Hz using `SAMPLE_INTERVAL_US = 4000` via `micros()`. Zero interrupt watchdog crashes. GPIO 2 heartbeat LED toggles every 500 ms for visual status confirmation.
- **Serial Transmission**: Verified live on `COM5` (Silicon Labs CP210x USB to UART Bridge). Streaming raw ADC values and lead-off flags without DTR/RTS bootloader lock.
- **Telemetry UI**: Waveform auto-scales dynamically; 10s warmup bar counts up from 0.0s to 10.0s; BPM ticker calculates estimated heart rate from detected peaks; diagnostic warning banner dynamically guides the user if packet rates drop to 0.0 Hz.
- **Training Metric Logging**: `train.py` computes and prints side-by-side training and validation metrics (`train_loss`, `train_acc`, `train_f1` vs `val_loss`, `val_acc`, `val_f1`), saving complete histories into model checkpoints.
