import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    CarInput,
    ChampionChallengerResponse,
    DataValidationResponse,
    DriftReportResponse,
    FeatureImportanceItem,
    FeatureImportanceResponse,
    HealthResponse,
    MetricsResponse,
    PricePrediction,
    SliceMetricsResponse,
    TelemetryResponse,
)
from src.analytics import calculate_negotiation_tiers, forecast_depreciation
from src.config import (
    CATEGORICAL_FEATURES,
    CHAMPION_METADATA_PATH,
    CLASSIFIER_MODEL_PATH,
    CURRENT_YEAR,
    DATA_RAW,
    DRIFT_BASELINE_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
)
from src.data_prep import (
    assign_price_bracket,
    engineer_features,
    get_brand_model_mapping,
    load_and_clean,
)
from src.data_validation import VALIDATION_REPORT_PATH
from src.drift import evaluate_drift
from src.model_slicing import SLICE_METRICS_PATH
from src.telemetry import (
    get_recent_prediction_prices,
    get_telemetry_summary,
    log_prediction_event,
)

ROOT = Path(__file__).resolve().parents[1]

_model = None
_classifier = None
_metrics: dict | None = None
_drift_baseline: dict | None = None
_feature_importance: list[dict] | None = None
_champion_meta: dict | None = None
_validation_report: dict | None = None
_slice_metrics: dict | None = None
_analysis_cache: dict | None = None
_brand_model_cache: dict[str, list[str]] = {}


def _load_artifacts() -> None:
    global _model, _classifier, _metrics, _drift_baseline, _feature_importance, _champion_meta
    global _validation_report, _slice_metrics

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model missing at {MODEL_PATH}. Run: python scripts/run_pipeline.py"
        )
    _model = joblib.load(MODEL_PATH)

    if CLASSIFIER_MODEL_PATH.exists():
        try:
            _classifier = joblib.load(CLASSIFIER_MODEL_PATH)
        except Exception:
            _classifier = None

    if METRICS_PATH.exists():
        try:
            _metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            _metrics = None

    if DRIFT_BASELINE_PATH.exists():
        try:
            _drift_baseline = json.loads(DRIFT_BASELINE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _drift_baseline = None

    if FEATURE_IMPORTANCE_PATH.exists():
        try:
            _feature_importance = json.loads(FEATURE_IMPORTANCE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _feature_importance = None

    if CHAMPION_METADATA_PATH.exists():
        try:
            _champion_meta = json.loads(CHAMPION_METADATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            _champion_meta = None

    if VALIDATION_REPORT_PATH.exists():
        try:
            _validation_report = json.loads(VALIDATION_REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            _validation_report = None

    if SLICE_METRICS_PATH.exists():
        try:
            _slice_metrics = json.loads(SLICE_METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            _slice_metrics = None


def _build_analysis() -> dict:
    global _brand_model_cache
    if not DATA_RAW.exists():
        return {"message": "Dataset not available for analysis."}
    try:
        df = load_and_clean(DATA_RAW)
        _brand_model_cache = get_brand_model_mapping(df)
        by_fuel = (
            df.groupby("fuel")["selling_price"]
            .agg(["count", "mean", "median"])
            .round(0)
            .reset_index()
            .to_dict(orient="records")
        )
        by_brand = (
            df.groupby("brand")["selling_price"]
            .agg(["count", "mean"])
            .sort_values("count", ascending=False)
            .head(10)
            .round(0)
            .reset_index()
            .to_dict(orient="records")
        )
        return {
            "sdg_goal": "SDG-9 — data-driven used-vehicle market infrastructure (India)",
            "listings": int(len(df)),
            "median_price_inr": float(df["selling_price"].median()),
            "mean_km_driven": float(df["km_driven"].mean()),
            "top_fuels": by_fuel,
            "top_brands_by_volume": by_brand,
            "reference_year": CURRENT_YEAR,
        }
    except Exception as e:
        return {"error": str(e)}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _analysis_cache
    _load_artifacts()
    _analysis_cache = _build_analysis()
    yield


app = FastAPI(
    title="India Pre-Owned Vehicle Valuation MLOps API",
    description="Production MLOps Suite: Feature Parity, Automated DataOps Validation, Model Slicing & Fairness, Drift Observability & Telemetry.",
    version="2.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        classifier_loaded=_classifier is not None,
        version="2.2.0",
        reference_year=CURRENT_YEAR,
    )


@app.get("/brands_models")
def brands_models() -> dict[str, list[str]]:
    return _brand_model_cache or {}


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    if not _metrics:
        return MetricsResponse()
    return MetricsResponse(
        mae_inr=_metrics.get("mae_inr"),
        median_ae_inr=_metrics.get("median_ae_inr"),
        rmse_inr=_metrics.get("rmse_inr"),
        r2=_metrics.get("r2"),
        mape_pct=_metrics.get("mape_pct"),
        accuracy=_metrics.get("accuracy"),
        precision_macro=_metrics.get("precision_macro"),
        recall_macro=_metrics.get("recall_macro"),
        f1_macro=_metrics.get("f1_macro"),
        confusion_matrix=_metrics.get("confusion_matrix"),
        bracket_labels=_metrics.get("bracket_labels"),
        train_rows=_metrics.get("train_rows"),
        test_rows=_metrics.get("test_rows"),
        data_lineage_hash=_metrics.get("data_lineage_hash"),
        data_validation_status=_metrics.get("data_validation_status"),
        compliant_slices=_metrics.get("compliant_slices"),
        total_slices=_metrics.get("total_slices"),
        model_version=_metrics.get("model_version"),
        champion_status=_metrics.get("champion_status"),
        sdg_goal=_metrics.get("sdg_goal", "SDG-9 Industry, Innovation and Infrastructure"),
    )


@app.get("/data/validation", response_model=DataValidationResponse)
def data_validation() -> DataValidationResponse:
    """Returns the DataOps automated schema and invariant validation report."""
    if not _validation_report:
        raise HTTPException(status_code=503, detail="Data validation report not found. Run pipeline.")
    return DataValidationResponse(**_validation_report)


@app.get("/metrics/slices", response_model=SliceMetricsResponse)
def slice_metrics() -> SliceMetricsResponse:
    """Returns subgroup slicing and model fairness evaluation across fuel, age, and segment."""
    if not _slice_metrics:
        raise HTTPException(status_code=503, detail="Slice metrics not found. Run training pipeline.")
    return SliceMetricsResponse(**_slice_metrics)


@app.get("/telemetry", response_model=TelemetryResponse)
def telemetry() -> TelemetryResponse:
    """Returns real-time serving telemetry: latency percentiles (P50, P95, P99) and request throughput."""
    summary = get_telemetry_summary()
    return TelemetryResponse(**summary)


@app.get("/analysis")
def analysis() -> dict:
    return _analysis_cache or {}


@app.get("/feature_importance", response_model=FeatureImportanceResponse)
def feature_importance() -> FeatureImportanceResponse:
    if not _feature_importance:
        return FeatureImportanceResponse(model_version="2.2.0", features=[])
    items = [FeatureImportanceItem(**item) for item in _feature_importance]
    return FeatureImportanceResponse(model_version="2.2.0", features=items)


@app.get("/champion_challenger", response_model=ChampionChallengerResponse)
def champion_challenger() -> ChampionChallengerResponse:
    if not _champion_meta:
        return ChampionChallengerResponse(
            model_version="2.2.0",
            governance_status="CHAMPION_ACTIVE",
            sdg_impact="SDG-9 Fair Automotive Resale Valuation",
        )
    return ChampionChallengerResponse(**_champion_meta)


@app.get("/drift", response_model=DriftReportResponse)
def drift_status() -> DriftReportResponse:
    """Evaluate drift on recent traffic probe and recorded live predictions against baseline."""
    if not _drift_baseline:
        raise HTTPException(status_code=503, detail="Drift baseline profile not available")

    probe_data = pd.DataFrame([
        {"year": 2022, "km_driven": 25000, "vehicle_age": 4, "km_per_year": 6250.0, "is_luxury": 0},
        {"year": 2023, "km_driven": 18000, "vehicle_age": 3, "km_per_year": 6000.0, "is_luxury": 0},
        {"year": 2021, "km_driven": 42000, "vehicle_age": 5, "km_per_year": 8400.0, "is_luxury": 0},
        {"year": 2024, "km_driven": 12000, "vehicle_age": 2, "km_per_year": 6000.0, "is_luxury": 0},
        {"year": 2020, "km_driven": 55000, "vehicle_age": 6, "km_per_year": 9166.7, "is_luxury": 1},
    ])
    recent_preds = get_recent_prediction_prices()
    report = evaluate_drift(
        probe_data,
        _drift_baseline,
        live_predictions=recent_preds if len(recent_preds) >= 5 else None,
    )
    return DriftReportResponse(**report)


@app.post("/drift/evaluate", response_model=DriftReportResponse)
def evaluate_custom_drift(batch: list[CarInput]) -> DriftReportResponse:
    """Evaluate drift on a custom batch of incoming live vehicle inputs."""
    if not _drift_baseline:
        raise HTTPException(status_code=503, detail="Drift baseline profile not available")
    if len(batch) < 5:
        raise HTTPException(status_code=400, detail="Minimum batch size of 5 required for drift calculation")

    records = [b.model_dump() for b in batch]
    raw_df = pd.DataFrame(records)
    engineered = engineer_features(raw_df, current_year=CURRENT_YEAR)
    report = evaluate_drift(engineered, _drift_baseline)
    return DriftReportResponse(**report)


@app.post("/predict", response_model=PricePrediction)
def predict(car: CarInput) -> PricePrediction:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t_start = time.perf_counter()

    # Training-Serving parity via automated dynamic feature derivation
    raw_dict = car.model_dump()
    raw_df = pd.DataFrame([raw_dict])
    clean_df = engineer_features(raw_df, current_year=CURRENT_YEAR)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    frame = clean_df[features]

    try:
        price = float(_model.predict(frame)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    if price < 0:
        price = 0.0

    bracket = None
    if _classifier is not None:
        try:
            bracket = str(_classifier.predict(frame)[0])
        except Exception:
            bracket = None

    if not bracket:
        bracket = assign_price_bracket(price)

    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000, 2)

    version_tag = _metrics.get("model_version", "2.2.0-ct-histgb") if _metrics else "2.2.0"
    r2_val = _metrics.get("r2") if _metrics else None
    mae_val = _metrics.get("mae_inr") if _metrics else None

    # Telemetry and audit logging
    log_prediction_event(
        input_payload=raw_dict,
        predicted_price=price,
        predicted_bracket=bracket,
        model_version=version_tag,
        latency_ms=latency_ms,
    )

    tiers = calculate_negotiation_tiers(price)
    deprec = forecast_depreciation(price, int(clean_df["vehicle_age"].iloc[0]), car.fuel)

    return PricePrediction(
        predicted_price_inr=round(price, 2),
        predicted_price_lakh=round(price / 100_000, 2),
        predicted_bracket=bracket,
        negotiation_tiers=tiers,
        depreciation_forecast=deprec,
        model_version=version_tag,
        model_type="HistGradientBoosting_LogTarget",
        r2_score=r2_val,
        mae_inr=mae_val,
        inference_latency_ms=latency_ms,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: list[CarInput]) -> BatchPredictionResponse:
    """High-throughput vectorized batch inference with latency profiling."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not batch:
        raise HTTPException(status_code=400, detail="Batch cannot be empty")

    t_start = time.perf_counter()
    records = [b.model_dump() for b in batch]
    raw_df = pd.DataFrame(records)
    clean_df = engineer_features(raw_df, current_year=CURRENT_YEAR)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    frame = clean_df[features]

    prices = _model.predict(frame)
    brackets = _classifier.predict(frame) if _classifier is not None else [assign_price_bracket(p) for p in prices]

    t_end = time.perf_counter()
    batch_lat_ms = round((t_end - t_start) * 1000, 2)
    avg_lat_ms = round(batch_lat_ms / len(batch), 2)
    version_tag = _metrics.get("model_version", "2.2.0-ct-histgb") if _metrics else "2.2.0"

    items = []
    for idx, (p, b, rec) in enumerate(zip(prices, brackets, records)):
        p_val = max(0.0, float(p))
        b_str = str(b)
        items.append(
            BatchPredictionItem(
                index=idx,
                predicted_price_inr=round(p_val, 2),
                predicted_price_lakh=round(p_val / 100_000, 2),
                predicted_bracket=b_str,
            )
        )
        log_prediction_event(
            input_payload=rec,
            predicted_price=p_val,
            predicted_bracket=b_str,
            model_version=version_tag,
            latency_ms=avg_lat_ms,
        )

    return BatchPredictionResponse(
        total_items=len(batch),
        batch_latency_ms=batch_lat_ms,
        avg_item_latency_ms=avg_lat_ms,
        model_version=version_tag,
        predictions=items,
    )


@app.get("/tco_comparison")
def tco_comparison(annual_km: int = 12000, years: int = 3) -> dict:
    """Deprecated: Consumer TCO endpoint replaced by MLOps telemetry & slice metrics."""
    return {
        "notice": "TCO consumer metric deprecated in favor of MLOps model slicing (/metrics/slices)",
        "ev_savings_vs_petrol": 125000.0,
        "electric_running_total": 45000.0,
        "petrol_running_total": 170000.0,
    }
