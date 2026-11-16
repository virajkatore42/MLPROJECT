import json
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "price_model.joblib"


class DirectAPIClient:
    """Zero-deadlock test client for endpoints across all Python versions and platforms."""
    def __init__(self):
        from app.main import (
            _build_analysis,
            _load_artifacts,
            brands_models,
            champion_challenger,
            data_validation,
            drift_status,
            feature_importance,
            health,
            metrics,
            predict,
            predict_batch,
            slice_metrics,
            tco_comparison,
            telemetry,
        )
        _load_artifacts()
        self._analysis = _build_analysis()
        self._health = health
        self._predict = predict
        self._predict_batch = predict_batch
        self._metrics = metrics
        self._drift = drift_status
        self._importance = feature_importance
        self._champion = champion_challenger
        self._tco = tco_comparison
        self._brands_models = brands_models
        self._data_validation = data_validation
        self._slice_metrics = slice_metrics
        self._telemetry = telemetry

    def get(self, path: str):
        class Response:
            def __init__(self, data, status_code=200):
                self._data = data
                self.status_code = status_code

            def json(self):
                if hasattr(self._data, "model_dump"):
                    return self._data.model_dump()
                return self._data

        if path == "/health":
            return Response(self._health())
        elif path == "/metrics":
            return Response(self._metrics())
        elif path == "/drift":
            return Response(self._drift())
        elif path == "/feature_importance":
            return Response(self._importance())
        elif path == "/champion_challenger":
            return Response(self._champion())
        elif path == "/brands_models":
            return Response(self._brands_models())
        elif path == "/data/validation":
            return Response(self._data_validation())
        elif path == "/metrics/slices":
            return Response(self._slice_metrics())
        elif path == "/telemetry":
            return Response(self._telemetry())
        elif "/tco_comparison" in path:
            return Response(self._tco())
        return Response({}, status_code=404)

    def post(self, path: str, json: dict | list):
        class Response:
            def __init__(self, data, status_code=200):
                self._data = data
                self.status_code = status_code

            def json(self):
                if hasattr(self._data, "model_dump"):
                    return self._data.model_dump()
                return self._data

        if path == "/predict":
            from app.schemas import CarInput
            inp = CarInput(**json)
            return Response(self._predict(inp))
        elif path == "/predict/batch":
            from app.schemas import CarInput
            inp_list = [CarInput(**item) for item in json]
            return Response(self._predict_batch(inp_list))
        return Response({}, status_code=404)


@pytest.fixture(scope="module")
def client():
    if not MODEL.exists():
        pytest.skip("Model not trained — run python -m src.train")
    if sys.platform == "win32" and sys.version_info >= (3, 14):
        return DirectAPIClient()
    try:
        from fastapi.testclient import TestClient
        from app.main import app
        with TestClient(app) as test_client:
            return test_client
    except Exception:
        return DirectAPIClient()


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["classifier_loaded"] is True
    assert body["reference_year"] >= 2026


def test_predict_ice(client):
    payload = {
        "year": 2022,
        "km_driven": 35000,
        "brand": "Maruti",
        "car_model": "Swift",
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["predicted_price_inr"] > 200_000
    assert body["predicted_price_lakh"] > 2.0
    assert body["predicted_bracket"] in ["Budget", "Mid-Range", "Premium", "Luxury"]
    assert "inference_latency_ms" in body
    assert body["inference_latency_ms"] >= 0.0


def test_predict_ev(client):
    payload = {
        "year": 2023,
        "km_driven": 18000,
        "brand": "Tata",
        "car_model": "Nexon-EV",
        "fuel": "Electric",
        "seller_type": "Individual",
        "transmission": "Automatic",
        "owner": "First Owner",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["predicted_price_inr"] > 500_000
    assert body["predicted_bracket"] in ["Mid-Range", "Premium", "Luxury"]


def test_predict_hybrid(client):
    payload = {
        "year": 2024,
        "km_driven": 15000,
        "brand": "Toyota",
        "car_model": "Urban-Cruiser-Hyryder",
        "fuel": "Hybrid",
        "seller_type": "Individual",
        "transmission": "Automatic",
        "owner": "First Owner",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["predicted_price_inr"] > 700_000


def test_batch_prediction_endpoint(client):
    batch = [
        {
            "year": 2023,
            "km_driven": 15000,
            "brand": "Hyundai",
            "car_model": "Creta",
            "fuel": "Diesel",
            "seller_type": "Individual",
            "transmission": "Manual",
            "owner": "First Owner",
        },
        {
            "year": 2024,
            "km_driven": 10000,
            "brand": "Tata",
            "car_model": "Nexon-EV",
            "fuel": "Electric",
            "seller_type": "Dealer",
            "transmission": "Automatic",
            "owner": "First Owner",
        },
    ]
    res = client.post("/predict/batch", json=batch)
    assert res.status_code == 200
    body = res.json()
    assert body["total_items"] == 2
    assert len(body["predictions"]) == 2
    assert body["batch_latency_ms"] >= 0.0


def test_drift_endpoint(client):
    res = client.get("/drift")
    assert res.status_code == 200
    body = res.json()
    assert "overall_status" in body
    assert "max_psi" in body
    assert body["live_samples"] >= 5


def test_feature_importance_endpoint(client):
    res = client.get("/feature_importance")
    assert res.status_code == 200
    body = res.json()
    assert "features" in body
    assert len(body["features"]) >= 5
    features = [f["feature"] for f in body["features"]]
    assert "vehicle_age" in features or "brand" in features


def test_champion_challenger_endpoint(client):
    res = client.get("/champion_challenger")
    assert res.status_code == 200
    body = res.json()
    assert "governance_status" in body
    assert "model_version" in body


def test_data_validation_endpoint(client):
    res = client.get("/data/validation")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "PASSED"
    assert body["total_checks"] >= 5
    assert body["pass_rate_pct"] == 100.0


def test_slice_metrics_endpoint(client):
    res = client.get("/metrics/slices")
    assert res.status_code == 200
    body = res.json()
    assert body["total_slices_evaluated"] >= 10
    assert body["compliant_slices"] >= 10
    dimensions = {s["dimension"] for s in body["slices"]}
    assert "Fuel / Powertrain" in dimensions
    assert "Vehicle Age Bracket" in dimensions


def test_telemetry_endpoint(client):
    res = client.get("/telemetry")
    assert res.status_code == 200
    body = res.json()
    assert "total_predictions_served" in body
    assert "latency_p50_ms" in body
    assert body["total_predictions_served"] >= 1


def test_brands_models(client):
    res = client.get("/brands_models")
    assert res.status_code == 200
    mapping = res.json()
    assert isinstance(mapping, dict)
    assert "Tata" in mapping
    assert any("EV" in m.upper() or "Nexon" in m for m in mapping["Tata"])
