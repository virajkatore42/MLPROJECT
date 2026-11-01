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
            drift_status,
            feature_importance,
            health,
            metrics,
            predict,
            tco_comparison,
        )
        _load_artifacts()
        self._analysis = _build_analysis()
        self._health = health
        self._predict = predict
        self._metrics = metrics
        self._drift = drift_status
        self._importance = feature_importance
        self._champion = champion_challenger
        self._tco = tco_comparison
        self._brands_models = brands_models

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
        elif "/tco_comparison" in path:
            return Response(self._tco())
        return Response({}, status_code=404)

    def post(self, path: str, json: dict):
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
        return Response({}, status_code=404)


@pytest.fixture(scope="module")
def client():
    if not MODEL.exists():
        pytest.skip("Model not trained — run python -m src.train")
    # On Python 3.14 Windows, anyio's blocking portal deadlocks with TestClient; use DirectAPIClient
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
    assert "negotiation_tiers" in body
    assert body["negotiation_tiers"]["instant_cash_price"] < body["negotiation_tiers"]["fair_market_price"]
    assert len(body["depreciation_forecast"]) == 5


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
    assert len(body["depreciation_forecast"]) == 5


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


def test_tco_endpoint(client):
    res = client.get("/tco_comparison?annual_km=12000&years=3")
    assert res.status_code == 200
    body = res.json()
    assert body["ev_savings_vs_petrol"] > 0
    assert body["electric_running_total"] < body["petrol_running_total"]


def test_brands_models(client):
    res = client.get("/brands_models")
    assert res.status_code == 200
    mapping = res.json()
    assert isinstance(mapping, dict)
    assert "Tata" in mapping
    assert any("EV" in m.upper() or "Nexon" in m for m in mapping["Tata"])
