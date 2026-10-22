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