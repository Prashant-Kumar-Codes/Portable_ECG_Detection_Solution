# Important Errors and Their Fixes (IEF)

---

## 1. Streaming Error: "No Incoming Bytes" Despite Connected Sensor

### Observed Symptoms
- The live dashboard opens at `http://127.0.0.1:8000`.
- The system health pill shows `Connected`, but the sidebar reports `No Incoming Bytes` in red.
- The 10-second baseline warmup bar stays frozen at `0.0s / 10.0s (0%)`.
- No waveform appears on the canvas.

### Root Causes
1. **ESP32 Firmware Fatal Panic (`analogRead` in ISR)**:
   In `firmware/esp32_ad8232_250hz/esp32_ad8232_250hz.ino`, `analogRead(ECG_PIN)` was executed inside `IRAM_ATTR onTimer()` within a critical section. On ESP32, `analogRead` uses FreeRTOS mutexes and ADC calibration routines that are **not IRAM-safe**. Calling it from a timer interrupt triggers a `Guru Meditation Error (Interrupt wdt timeout / LoadProhibited)`, causing the ESP32 to crash and reboot in a continuous loop without transmitting serial packets.
2. **PySerial DTR/RTS Line Lock**:
   On Windows, opening a COM port via `pyserial` asserts DTR and RTS by default. On ESP32 development boards (DevKit, NodeMCU with CH340/CP2102), DTR and RTS are wired to the auto-reset transistors connected to `EN` (Reset) and `IO0` (Boot). If not explicitly cleared (`ser.dtr = False; ser.rts = False`), the ESP32 is held in hardware reset or ROM bootloader download mode.
3. **Bluetooth Virtual COM Port Trap (`COM3` / `COM4`)**:
   On Windows laptops, `COM3` and `COM4` are commonly virtual Bluetooth ports (`Standard Serial over Bluetooth link`). Windows allows `serial.Serial("COM3")` to open without throwing an exception. However, because no Bluetooth device is actually streaming, `readline()` times out continuously, resulting in 0 bytes received.
4. **Physical USB Cable / Windows PnP Code 45**:
   If a charge-only USB cable is used, the ESP32 and AD8232 power LEDs illuminate (giving the appearance of being properly connected), but the D+/D- data lines are physically absent. Windows reports the USB-to-UART bridge as `Problem 45: Device not connected`.

### Solutions Applied
1. **Firmware Fix (`esp32_ad8232_250hz.ino`)**:
   - Replaced the timer ISR with a rock-solid, non-blocking `micros()` 250 Hz loop (`SAMPLE_INTERVAL_US = 4000`).
   - `analogRead()` and `digitalRead()` now run safely inside `loop()` in user space with zero risk of interrupt watchdog panics.
   - Added a built-in LED blink on GPIO 2 (toggles every 500 ms) so the user can immediately confirm that the microcontroller firmware is active and running.
   - Compatible across all ESP32 core versions (v1, v2, v3) as well as Arduino Uno/Nano.
2. **Driver Fix (`Modules/serial_reader.py`)**:
   - Explicitly configured `ser.dtr = False; ser.rts = False` and added `ser.reset_input_buffer()` upon opening the serial port.
   - Made `hardware_connected` reflect real byte reception: if no valid packets arrive for 3 seconds, `hardware_connected` dynamically becomes `False` and an actionable diagnostic warning is logged.
3. **Port Auto-Detection & Link Warning (`run_live.py`)**:
   - Prioritizes real USB serial controllers (`CH340`, `CP210`, `FTDI`, `Arduino`, `ESP32`) during auto-detection.
   - Issues an explicit warning in console if a Bluetooth virtual port is selected.
4. **Dashboard Health Diagnostic (`static/index.html`)**:
   - Added a troubleshooting hint box that dynamically appears whenever "No Incoming Bytes" is detected, instructing the user to check cable data support, Device Manager, and baud rate.

---

## 2. Model Checkpoint Not Updating Despite Higher Accuracy/F1 & Early Stopping Not Triggering

### Observed Symptoms
- Training runs past patience limit (e.g., reaching Epoch 10 even when `EARLY_STOP_PATIENCE = 3`).
- Better classification performance (e.g., higher `val_acc` and `val_f1` at Epoch 4: 0.7748 and Epoch 8: 0.7959) does not trigger checkpoint saves.

### Root Causes
1. **Loss vs. Score Discrepancy**: Checkpointing in `train.py` was hardcoded to `val_metrics["loss"] < best_val_loss`. Cross-entropy loss on imbalanced multi-class data frequently increases as the model becomes overconfident or heavily penalized on rare classes, even while discrete decision boundaries (`val_acc` and `val_macro_f1`) continue to improve. Since `val_loss` at Epochs 4-10 (>0.50) was higher than Epoch 2 (0.4943), no subsequent checkpoint was saved.
2. **Jupyter Module Caching & Function Default Binding**: In Jupyter, `import config` is cached in `sys.modules`. Modifying `config.py` on disk does not update values in memory without a kernel restart or `%autoreload 2`. Furthermore, `Trainer.__init__` bound default arguments at module definition time.

### Solutions Applied
1. **Configurable Metric Monitoring (`Modules/train.py` & `Modules/config.py`)**:
   - Added `MONITOR_METRIC` (default: `"val_f1"`, or optionally `"val_loss"`, `"val_acc"`).
   - Checkpointing and early stopping now track the designated metric (mode `"max"` for Macro-F1 / Accuracy, mode `"min"` for Loss).
   - Initialization logging explicitly outputs `patience`, `max_epochs`, and `monitor (mode)` for full transparency.
2. **Jupyter Interactive Workflow**:
   - Use `%load_ext autoreload` + `%autoreload 2` in Notebook Cell 1 or pass `patience=config.EARLY_STOP_PATIENCE, monitor=config.MONITOR_METRIC` directly into `Trainer(...)`.

---