"""MLOps Production Serving Telemetry & Prediction Audit Logger.
Tracks real-time operational serving metrics:
- Latency percentiles (P50, P95, P99 in milliseconds)
- Throughput and total requests served
- Prediction distribution tracking for downstream concept drift
- Persistent JSONL audit logging for governance & traceability
"""

import json
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PROJECT_ROOT

AUDIT_LOG_DIR = PROJECT_ROOT / "data" / "telemetry"
AUDIT_LOG_PATH = AUDIT_LOG_DIR / "prediction_audit.jsonl"

_MAX_BUFFER_SIZE = 1000
_prediction_buffer: deque[dict[str, Any]] = deque(maxlen=_MAX_BUFFER_SIZE)
_service_start_time = time.time()


def log_prediction_event(
    input_payload: dict[str, Any],
    predicted_price: float,
    predicted_bracket: str,
    model_version: str,
    latency_ms: float,
    status: str = "SUCCESS",
) -> dict[str, Any]:
    """Logs a single inference transaction to memory buffer and disk audit log."""
    event = {
        "request_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "latency_ms": round(latency_ms, 2),
        "status": status,
        "model_version": model_version,
        "predicted_price_inr": round(predicted_price, 2),
        "predicted_bracket": predicted_bracket,
        "input_features": input_payload,
    }

    _prediction_buffer.append(event)

    # Persist to disk
    try:
        AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass

    return event


def get_telemetry_summary() -> dict[str, Any]:
    """Calculate real-time operational serving telemetry."""
    uptime_sec = round(time.time() - _service_start_time, 1)
    total_events = len(_prediction_buffer)

    if total_events == 0:
        return {
            "uptime_seconds": uptime_sec,
            "total_predictions_served": 0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_p99_ms": 0.0,
            "avg_latency_ms": 0.0,
            "avg_predicted_price_inr": 0.0,
            "recent_events": [],
            "status": "HEALTHY_IDLE",
        }

    latencies = [e["latency_ms"] for e in _prediction_buffer]
    prices = [e["predicted_price_inr"] for e in _prediction_buffer]

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    avg_lat = float(np.mean(latencies))
    avg_price = float(np.mean(prices))

    recent = list(_prediction_buffer)[-10:]
    recent.reverse()

    return {
        "uptime_seconds": uptime_sec,
        "total_predictions_served": total_events,
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "avg_latency_ms": round(avg_lat, 2),
        "avg_predicted_price_inr": round(avg_price, 0),
        "recent_events": recent,
        "status": "HEALTHY_ACTIVE",
    }


def get_recent_prediction_prices() -> list[float]:
    """Retrieve recorded predictions for concept / prediction drift computation."""
    return [e["predicted_price_inr"] for e in _prediction_buffer]
