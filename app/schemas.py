from typing import Any, Literal
from pydantic import BaseModel, Field

Fuel = Literal["Petrol", "Diesel", "CNG", "LPG", "Electric", "Hybrid"]
Seller = Literal["Individual", "Dealer", "Trustmark Dealer"]
Transmission = Literal["Manual", "Automatic"]
Owner = Literal[
    "First Owner",
    "Second Owner",
    "Third Owner",
    "Fourth Owner",
    "Fourth & Above Owner",
    "Test Drive Car",
]


class CarInput(BaseModel):
    year: int = Field(..., ge=1990, le=2030, examples=[2022])
    km_driven: int = Field(..., ge=0, le=500_000, examples=[28000])
    brand: str = Field(..., min_length=1, max_length=64, examples=["Tata"])
    car_model: str = Field("Other", max_length=64, examples=["Nexon-EV"])
    fuel: Fuel = "Petrol"
    seller_type: Seller = "Individual"
    transmission: Transmission = "Manual"
    owner: Owner = "First Owner"


class NegotiationTiers(BaseModel):
    instant_cash_price: float
    fair_market_price: float
    patient_seller_price: float
    dealer_margin_gap: float


class DepreciationPoint(BaseModel):
    period: str
    months_ahead: int
    projected_price_inr: float
    projected_price_lakh: float
    value_retention_pct: float


class PricePrediction(BaseModel):
    predicted_price_inr: float
    predicted_price_lakh: float
    predicted_bracket: str = "Mid-Range"
    negotiation_tiers: NegotiationTiers | None = None
    depreciation_forecast: list[DepreciationPoint] | None = None
    model_version: str = "2.1.0-ct-ensemble"
    model_type: str = "HistGradientBoosting_LogTarget"
    r2_score: float | None = None
    mae_inr: float | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classifier_loaded: bool
    version: str = "2.1.0"
    reference_year: int = 2026


class MetricsResponse(BaseModel):
    mae_inr: float | None = None
    median_ae_inr: float | None = None
    rmse_inr: float | None = None
    r2: float | None = None
    mape_pct: float | None = None
    accuracy: float | None = None
    precision_macro: float | None = None
    recall_macro: float | None = None
    f1_macro: float | None = None
    confusion_matrix: list[list[int]] | None = None
    bracket_labels: list[str] | None = None
    train_rows: int | None = None
    test_rows: int | None = None
    data_lineage_hash: str | None = None
    model_version: str | None = None
    champion_status: str | None = None
    sdg_goal: str = "SDG-9 Industry, Innovation and Infrastructure"


class FeatureImportanceItem(BaseModel):
    feature: str
    importance_pct: float
    category: str


class FeatureImportanceResponse(BaseModel):
    model_version: str = "2.1.0"
    features: list[FeatureImportanceItem]


class DriftMetricDetail(BaseModel):
    psi: float
    ks_statistic: float
    status: str
    live_mean: float
    baseline_mean: float


class DriftReportResponse(BaseModel):
    overall_status: str
    drift_detected: bool
    retraining_recommended: bool
    max_psi: float
    average_psi: float | None = None
    live_samples: int
    feature_metrics: dict[str, DriftMetricDetail]


class ChampionChallengerResponse(BaseModel):
    model_version: str
    governance_status: str
    promoted_at: str | None = None
    mae_inr: float | None = None
    r2_score: float | None = None
    accuracy: float | None = None
    previous_mae_inr: float | None = None
    previous_r2_score: float | None = None
    sdg_impact: str
