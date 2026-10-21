"""
server.py
---------
FastAPI application for real-time ECG arrhythmia monitoring.

Endpoints
---------
  GET  /          Serves static/index.html (live dashboard)
  GET  /status    System status JSON
  WS   /ws/ecg   Real-time ECG sample + prediction stream

WebSocket message schema (JSON, server -> client):
  {
    "ecg":        [float, ...]  - raw ADC samples (BATCH_SIZE per message)
    "lead_off":   bool          - electrode disconnected flag
    "prediction": {             - null until first inference fires
      "class_id":      int,
      "class_name":    str,
      "confidence":    float,
      "probabilities": [float x 5]
    } | null
  }
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from config import BEST_MODEL_PATH, CLASS_NAMES, STRIDE_SAMPLES
from inference import InferenceEngine
from serial_reader import BATCH_SIZE, SerialReader

logger = logging.getLogger(__name__)

# Absolute path to static/index.html (project_root/static/index.html)
_STATIC_HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"

# Run inference every STRIDE_SAMPLES new samples (500 = every 2 seconds)
_INFERENCE_EVERY = STRIDE_SAMPLES


def create_app(
    serial_port: Optional[str] = None,
    baud: int = 115_200,
    model_path: Path = BEST_MODEL_PATH,
    device: str = "cpu",
) -> FastAPI:
    """Construct and return the configured FastAPI application.

    Parameters
    ----------
    serial_port : str, optional
        Serial port for the ESP32 (e.g. 'COM3'). None activates mock mode.
    baud : int
        UART baud rate. Default 115200.
    model_path : Path
        Path to the trained .pt checkpoint.
    device : str
        PyTorch device ('cpu', 'cuda', 'cuda:0').
    """
    app = FastAPI(title="ECG Arrhythmia Monitor")

    reader = SerialReader(port=serial_port, baud=baud)
    engine = InferenceEngine(model_path=model_path, device=device)

    # ------------------------------------------------------------------ Lifecycle

    @app.on_event("startup")
    async def _startup():
        reader.start()
        logger.info("ECG monitor server started.")

    @app.on_event("shutdown")
    async def _shutdown():
        reader.stop()
        logger.info("ECG monitor server stopped.")

    # ------------------------------------------------------------------ HTTP

    @app.get("/", response_class=HTMLResponse)
    async def index():
        if not _STATIC_HTML.exists():
            return HTMLResponse(
                "<h1>Dashboard not found</h1><p>Missing static/index.html</p>",
                status_code=500,
            )
        return HTMLResponse(_STATIC_HTML.read_text(encoding="utf-8"))

    @app.get("/status")
    async def status():
        return JSONResponse({
            "mode":               "mock" if reader._mock else "serial",
            "port":               reader.port,
            "baud":               reader.baud,
            "hardware_connected": reader.hardware_connected,
            "lead_off":           reader.lead_off,
            "samples_received":   reader.samples_received,
            "sample_rate":        round(reader.sample_rate, 1),
            "is_ready":           reader.is_buffer_ready,
            "window_samples":     2500,
            "window_sec":         10.0,
            "model":              str(model_path.name),
            "model_path":         str(model_path),
            "classes":            CLASS_NAMES,
        })

    # ------------------------------------------------------------------ WebSocket

    @app.websocket("/ws/ecg")
    async def ws_ecg(websocket: WebSocket):
        await websocket.accept()
        logger.info("WebSocket client connected.")

        last_prediction = None
        pending = 0  # samples received since last inference
        last_heartbeat = 0.0

        try:
            import time
            while True:
                new_samples = reader.pop_new_samples()
                now = time.time()

                if new_samples:
                    pending += len(new_samples)

                    # Trigger inference every _INFERENCE_EVERY new samples once 10s buffer ready
                    if pending >= _INFERENCE_EVERY and reader.is_buffer_ready:
                        pending = 0
                        window = reader.get_window()
                        try:
                            last_prediction = engine.predict(window, lead_off=reader.lead_off)
                        except Exception as exc:
                            logger.warning(f"Inference error: {exc}")

                    # Status payload sent with updates
                    status_info = {
                        "hardware_connected": reader.hardware_connected,
                        "mode":               "mock" if reader._mock else "serial",
                        "port":               reader.port,
                        "baud":               reader.baud,
                        "sample_rate":        round(reader.sample_rate, 1),
                        "samples_received":   reader.samples_received,
                        "is_ready":           reader.is_buffer_ready,
                        "progress_pct":       min(100.0, round(reader.samples_received / 2500.0 * 100, 1)),
                        "progress_sec":       min(10.0, round(reader.samples_received / 250.0, 1)),
                    }

                    # Send samples in BATCH_SIZE chunks for smooth rendering
                    for i in range(0, len(new_samples), BATCH_SIZE):
                        chunk = new_samples[i : i + BATCH_SIZE]
                        await websocket.send_text(json.dumps({
                            "ecg":           chunk,
                            "lead_off":      reader.lead_off,
                            "prediction":    last_prediction,
                            "system_status": status_info,
                        }))
                    last_heartbeat = now
                else:
                    # Heartbeat if no samples arriving (e.g. idle or serial waiting)
                    if now - last_heartbeat >= 0.5:
                        status_info = {
                            "hardware_connected": reader.hardware_connected,
                            "mode":               "mock" if reader._mock else "serial",
                            "port":               reader.port,
                            "baud":               reader.baud,
                            "sample_rate":        round(reader.sample_rate, 1),
                            "samples_received":   reader.samples_received,
                            "is_ready":           reader.is_buffer_ready,
                            "progress_pct":       min(100.0, round(reader.samples_received / 2500.0 * 100, 1)),
                            "progress_sec":       min(10.0, round(reader.samples_received / 250.0, 1)),
                        }
                        await websocket.send_text(json.dumps({
                            "ecg":           [],
                            "lead_off":      reader.lead_off,
                            "prediction":    last_prediction,
                            "system_status": status_info,
                        }))
                        last_heartbeat = now

                # 50 ms sleep keeps UI responsive while maintaining low CPU usage
                await asyncio.sleep(0.05)

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected.")
        except Exception as exc:
            logger.error(f"WebSocket error: {exc}")
            try:
                await websocket.close()
            except Exception:
                pass

    return app
