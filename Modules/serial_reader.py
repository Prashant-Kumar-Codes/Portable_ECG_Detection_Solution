"""
serial_reader.py
----------------
Thread-safe ECG sample buffer backed by either a real ESP32 USB serial port
or a built-in synthetic mock generator (for testing without hardware).

ESP32 Serial Protocol (115200 baud, newline-delimited ASCII)
------------------------------------------------------------
  Plain int : "2048\n"              -> ADC sample, no lead-off
  JSON      : {"v":2048,"lo":0}\n  -> ADC sample + lead-off status

Ring Buffer
-----------
  Capacity: WINDOW_SAMPLES (2500) samples — always-full circular deque.
  Pre-filled with zeros so the inference engine can start immediately.

Mock Mode
---------
  Activated when port=None. A background thread generates a synthetic
  Lead-II-like waveform at 250 Hz for UI testing without the ESP32.
"""

import json
import logging
import math
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

WINDOW_SAMPLES = 2500   # 10 sec @ 250 Hz
BATCH_SIZE     = 25     # samples per WebSocket push -> 10 pushes/sec


def _synthetic_ecg_sample(t: float) -> float:
    """Return one synthetic ECG sample at time t (seconds).

    Approximates Lead-II morphology at ~75 bpm (0.8 s cycle).
    Not physiologically accurate — for UI testing only.
    """
    cycle = t % 0.8
    # P-wave
    p = 0.25 * math.exp(-((cycle - 0.05) ** 2) / (2 * 0.01 ** 2))
    # QRS complex (Q dip, dominant R spike, S dip)
    qrs = (
        -0.3 * math.exp(-((cycle - 0.18) ** 2) / (2 * 0.005 ** 2))
        + 1.5 * math.exp(-((cycle - 0.22) ** 2) / (2 * 0.005 ** 2))
        - 0.3 * math.exp(-((cycle - 0.26) ** 2) / (2 * 0.005 ** 2))
    )
    # T-wave
    t_wave = 0.35 * math.exp(-((cycle - 0.45) ** 2) / (2 * 0.03 ** 2))
    # Baseline wander + small noise
    noise = 0.02 * math.sin(2 * math.pi * 0.1 * t)
    return p + qrs + t_wave + noise


class SerialReader:
    """Thread-safe ECG acquisition buffer.

    Parameters
    ----------
    port : str, optional
        Serial port name ('COM3', '/dev/ttyUSB0', ...).
        If None, activates mock mode with a synthetic waveform.
    baud : int
        UART baud rate. Default 115200.
    """

    def __init__(self, port: Optional[str] = None, baud: int = 115_200):
        self._buffer: deque = deque(maxlen=WINDOW_SAMPLES)
        self._new_samples: list = []
        self._lock     = threading.Lock()
        self._lead_off = False
        self._running  = False
        self._thread: Optional[threading.Thread] = None
        self._samples_received = 0       # total samples received (for ready check)

        self.port = port
        self.baud = baud
        self._mock = (port is None)

        # Pre-fill with zeros so get_window() always returns WINDOW_SAMPLES values
        self._buffer.extend([0.0] * WINDOW_SAMPLES)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the background acquisition thread."""
        self._running = True
        target = self._mock_loop if self._mock else self._serial_loop
        self._thread = threading.Thread(target=target, daemon=True, name="SerialReader")
        self._thread.start()
        mode = "MOCK" if self._mock else f"SERIAL({self.port} @ {self.baud})"
        logger.info(f"SerialReader started in {mode} mode.")

    def stop(self):
        """Signal the acquisition thread to stop and wait for it."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("SerialReader stopped.")

    # ------------------------------------------------------------------
    # Public data accessors
    # ------------------------------------------------------------------

    def get_window(self) -> np.ndarray:
        """Return the current 2500-sample rolling window as float32 ndarray."""
        with self._lock:
            return np.array(self._buffer, dtype=np.float32)

    def pop_new_samples(self) -> list:
        """Return and clear all samples received since the last call.

        Returns
        -------
        list of float — raw ADC values (may be empty if called too frequently).
        """
        with self._lock:
            batch = list(self._new_samples)
            self._new_samples.clear()
            return batch

    @property
    def lead_off(self) -> bool:
        """True when the ESP32 signals a lead-off (electrode disconnected)."""
        return self._lead_off

    @property
    def is_buffer_ready(self) -> bool:
        """True once at least WINDOW_SAMPLES real samples have been received."""
        return self._samples_received >= WINDOW_SAMPLES

    # ------------------------------------------------------------------
    # Internal — push
    # ------------------------------------------------------------------

    def _push(self, value: float, lead_off: bool = False):
        with self._lock:
            self._buffer.append(value)
            self._new_samples.append(value)
            self._lead_off = lead_off
            self._samples_received += 1

    # ------------------------------------------------------------------
    # Acquisition loops (run in daemon thread)
    # ------------------------------------------------------------------

    def _serial_loop(self):
        """Read newline-delimited frames from the ESP32 over PySerial."""
        try:
            import serial
        except ImportError:
            logger.error("pyserial not installed. Install with: pip install pyserial")
            self._mock = True
            self._mock_loop()
            return

        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
            logger.info(f"Serial port {self.port} opened.")
        except Exception as exc:
            logger.error(f"Cannot open {self.port}: {exc}. Switching to mock mode.")
            self._mock = True
            self._mock_loop()
            return

        while self._running:
            try:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                value, lo = self._parse_line(line)
                self._push(value, lo)
            except Exception as exc:
                logger.warning(f"Serial read error: {exc}")

        ser.close()

    def _mock_loop(self):
        """Generate synthetic ECG at exactly 250 Hz."""
        t  = 0.0
        dt = 1.0 / 250.0
        while self._running:
            sample = _synthetic_ecg_sample(t) * 512 + 2048   # scale to ADC-like range
            self._push(sample, lead_off=False)
            t += dt
            time.sleep(dt)

    @staticmethod
    def _parse_line(line: str) -> tuple:
        """Parse a serial line into (adc_value: float, lead_off: bool)."""
        if line.startswith("{"):
            data = json.loads(line)
            return float(data.get("v", 0)), bool(data.get("lo", 0))
        return float(line), False
