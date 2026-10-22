
"""
run_live.py
-----------
Single entry point for the
real-time ECG arrhythmia monitor.

After training the model with Notebooks/N5_Training.ipynb, run this script
to start the live acquisition server and open the dashboard in your browser.

Usage
-----
  # Auto-detect microcontroller (or fallback to mock waveform):
  python run_live.py

  # Auto-scan specifically:
  python run_live.py --port auto

  # With specific USB port on Windows (e.g. COM5, COM6):
  python run_live.py --port COM5

  # With ESP32 on Linux/macOS:
  python run_live.py --port /dev/ttyUSB0

What it does
------------
1. Adds Modules/ to sys.path so server.py can import config, model, etc.
2. Loads the trained ECGCNNBiLSTM checkpoint from Dataset/Output_Data/saved_models/.
3. Starts a serial reader (real ESP32 or mock waveform generator).
4. Launches a FastAPI + WebSocket server on http://localhost:<http-port>.
5. Opens http://localhost:<http-port> in your default browser.
"""

import argparse
import logging
import os
import sys
import webbrowser
from pathlib import Path
from typing import Optional

# ----------------------------------------------------------------- sys.path
# Modules/ uses bare imports (from config import ...) so add it to the path.
PROJECT_ROOT = Path(__file__).resolve().parent
MODULES_DIR  = PROJECT_ROOT / "Modules"
if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))

# ----------------------------------------------------------------- Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_live")


# ----------------------------------------------------------------- CLI args
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time ECG Arrhythmia Monitor — starts the FastAPI server and opens the dashboard."
    )
    p.add_argument(
        "--port", "-p",
        default=None,
        help="Serial port of the ESP32 (e.g. COM3, /dev/ttyUSB0). "
             "Omit to run in mock mode with a synthetic waveform.",
    )
    p.add_argument(
        "--baud", "-b",
        type=int,
        default=115_200,
        help="Serial baud rate (default: 115200).",
    )
    p.add_argument(
        "--device", "-d",
        default="cpu",
        help="PyTorch device for inference ('cpu', 'cuda', 'cuda:0'). Default: cpu.",
    )
    p.add_argument(
        "--http-port",
        type=int,
        default=8000,
        help="HTTP/WebSocket server port (default: 8000).",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server bind address (default: 127.0.0.1). Use 0.0.0.0 for LAN access.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Path to a custom .pt checkpoint. "
             "Defaults to Dataset/Output_Data/saved_models/best_ecg_cnn_bilstm.pt.",
    )
    return p.parse_args()


# ----------------------------------------------------------------- Entry
def main():
    args = parse_args()

    # Lazy imports after sys.path is set
    from config import BEST_MODEL_PATH
    from server import create_app

    model_path = Path(args.model) if args.model else BEST_MODEL_PATH

    if not model_path.exists():
        logger.error(
            f"Model checkpoint not found: {model_path}\n"
            "Train the model first by running Notebooks/N5_Training.ipynb."
        )
        sys.exit(1)

    def _get_free_port(host: str, preferred_port: int) -> int:
        import socket
        for p in range(preferred_port, preferred_port + 20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((host, p))
                    return p
                except OSError:
                    continue
        return preferred_port

    def resolve_serial_port(requested_port: Optional[str]) -> Optional[str]:
        """Validate requested serial port or auto-detect microcontroller."""
        try:
            import serial.tools.list_ports
            available = list(serial.tools.list_ports.comports())
        except Exception:
            return requested_port

        # Auto-check for USB microcontroller if user omitted --port
        if not requested_port:
            for p in available:
                desc = (p.description or "").lower()
                hwid = (p.hwid or "").lower()
                if any(k in desc or k in hwid for k in ["ch340", "cp210", "ftdi", "usb serial", "arduino", "esp32"]):
                    logger.info(f"Auto-detected USB microcontroller on {p.device}: {p.description}")
                    return p.device
            return None

        req = requested_port.strip().upper()
        port_map = {p.device.upper(): p for p in available}

        # 1. Exact match (e.g. user passed COM5 and COM5 exists)
        if req in port_map:
            desc = port_map[req].description or ""
            if "bluetooth" in desc.lower():
                logger.warning("=" * 60)
                logger.warning(f"⚠️  NOTICE: '{req}' is a Bluetooth link: {desc}")
                logger.warning("  Windows Bluetooth virtual COM ports can open without error but send")
                logger.warning("  NO BYTES unless an active Bluetooth SPP stream is connected.")
                logger.warning("  If your ESP32 is plugged in with a USB cable, check your cable or Device Manager.")
                logger.warning("=" * 60)
            logger.info(f"Target port {req} selected: {desc}")
            return req

        # 2. Auto-detection mode (--port auto)
        if req == "AUTO":
            for p in available:
                desc = (p.description or "").lower()
                hwid = (p.hwid or "").lower()
                if any(k in desc or k in hwid for k in ["ch340", "cp210", "ftdi", "usb serial", "arduino", "esp32"]):
                    logger.info(f"Auto-detected microcontroller: {p.device} ({p.description})")
                    return p.device

            # Try any non-bluetooth port
            non_bt = [p for p in available if "bluetooth" not in (p.description or "").lower()]
            if non_bt:
                logger.info(f"Auto-selected serial device: {non_bt[0].device} ({non_bt[0].description})")
                return non_bt[0].device

            logger.warning("=" * 60)
            logger.warning("⚠️  NO HARDWARE DETECTED: No USB serial microcontroller found.")
            if available:
                logger.warning("  Detected ports (Bluetooth only):")
                for p in available:
                    logger.warning(f"    - {p.device} ({p.description})")
            else:
                logger.warning("  No active COM ports detected by Windows.")
            logger.warning("  Please check that the ESP32/Arduino USB cable is plugged into your PC.")
            logger.warning("  Falling back to MOCK MODE.")
            logger.warning("=" * 60)
            return None

        # 3. User requested a specific port that was not found
        port_list = [f"{p.device} ({p.description})" for p in available]
        logger.warning("=" * 60)
        logger.warning(f"⚠️  PORT NOT FOUND: '{requested_port}' is not currently detected by Windows!")

        # Check if another USB-Serial device is available
        for p in available:
            desc = (p.description or "").lower()
            if any(k in desc for k in ["ch340", "cp210", "ftdi", "usb serial", "arduino", "esp32"]):
                logger.warning(f"  --> Found alternative USB device: {p.device} ({p.description})! Connecting to {p.device}...")
                return p.device

        if available:
            logger.warning("  Currently connected COM ports:")
            for p_str in port_list:
                logger.warning(f"    - {p_str}")
        else:
            logger.warning("  No active COM ports detected by Windows.")

        logger.warning("  Troubleshooting steps:")
        logger.warning("    1. Is the USB cable firmly plugged into your computer and board?")
        logger.warning("    2. Is it a data cable (not charge-only)?")
        logger.warning("    3. Check Windows Device Manager under 'Ports (COM & LPT)'.")
        logger.warning("  Falling back to MOCK MODE so dashboard stays accessible.")
        logger.warning("=" * 60)
        return None

    chosen_port = resolve_serial_port(args.port)

    actual_port = _get_free_port(args.host, args.http_port)
    if actual_port != args.http_port:
        logger.warning(f"Port {args.http_port} is busy. Switched to available port {actual_port}.")

    logger.info("=" * 60)
    logger.info("  ECG Arrhythmia Monitor — Real-Time Inference Server")
    logger.info("=" * 60)
    logger.info(f"  Serial port : {chosen_port or 'MOCK MODE (no hardware)'}")
    logger.info(f"  Baud rate   : {args.baud}")
    logger.info(f"  Device      : {args.device}")
    logger.info(f"  Model       : {model_path}")
    logger.info(f"  Dashboard   : http://{args.host}:{actual_port}")
    logger.info("=" * 60)

    app = create_app(
        serial_port=chosen_port,
        baud=args.baud,
        model_path=model_path,
        device=args.device,
    )

    # Open browser after a short delay so the server has time to start
    dashboard_url = f"http://127.0.0.1:{actual_port}"
    import threading, time
    def _open_browser():
        time.sleep(1.5)
        webbrowser.open(dashboard_url)
    threading.Thread(target=_open_browser, daemon=True).start()

    # Start Uvicorn (FastAPI's ASGI server)
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn[standard]")
        sys.exit(1)

    uvicorn.run(
        app,
        host=args.host,
        port=actual_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
