#!/usr/bin/env python3
"""
run_live.py
-----------
Single entry point for the real-time ECG arrhythmia monitor.

After training the model with Notebooks/N5_Training.ipynb, run this script
to start the live acquisition server and open the dashboard in your browser.

Usage
-----
  # No hardware (mock waveform):
  python run_live.py

  # With ESP32 on Windows:
  python run_live.py --port COM3

  # With ESP32 on Linux/macOS:
  python run_live.py --port /dev/ttyUSB0

  # Custom options:
  python run_live.py --port COM3 --baud 115200 --device cuda --http-port 8000

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

    logger.info("=" * 60)
    logger.info("  ECG Arrhythmia Monitor — Real-Time Inference Server")
    logger.info("=" * 60)
    logger.info(f"  Serial port : {args.port or 'MOCK MODE (no hardware)'}")
    logger.info(f"  Baud rate   : {args.baud}")
    logger.info(f"  Device      : {args.device}")
    logger.info(f"  Model       : {model_path}")
    logger.info(f"  Dashboard   : http://{args.host}:{args.http_port}")
    logger.info("=" * 60)

    app = create_app(
        serial_port=args.port,
        baud=args.baud,
        model_path=model_path,
        device=args.device,
    )

    # Open browser after a short delay so the server has time to start
    dashboard_url = f"http://127.0.0.1:{args.http_port}"
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
        port=args.http_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
