import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "cardekho.csv"
DATA_RECENT = ROOT / "data" / "raw" / "recent_cars_2022_2026.csv"
METRICS_PATH = ROOT / "models" / "metrics.json"
DRIFT_PATH = ROOT / "models" / "drift_baseline.json"
FEATURE_IMP_PATH = ROOT / "models" / "feature_importance.json"
CHAMPION_PATH = ROOT / "models" / "champion_metadata.json"
VALIDATION_PATH = ROOT / "models" / "data_validation_report.json"
SLICE_PATH = ROOT / "models" / "slice_metrics.json"


def test_dataset_exists():
    assert DATA.exists(), "Cardekho dataset missing. Run python scripts/download_data.py"
    assert DATA_RECENT.exists(), "Recent 2021-2026 dataset missing. Run python scripts/build_recent_market_dataset.py"


def test_load_and_clean():
    from src.data_prep import load_and_clean

    df = load_and_clean(str(DATA))
    assert len(df) > 5000, f"Expected >5000 records, got {len(df)}"
    assert "selling_price" in df.columns
    assert "vehicle_age" in df.columns
    assert "km_per_year" in df.columns
    assert "is_luxury" in df.columns
    assert df["year"].max() >= 2024, "Dataset must include modern recent cars"


def test_data_invariants():
    from src.data_prep import load_and_clean

    df = load_and_clean(str(DATA))
    assert (df["selling_price"] > 0).all()
    assert (df["km_driven"] >= 0).all()
    assert (df["vehicle_age"] >= 0).all()
    assert not df["selling_price"].isna().any()
    assert not df["km_per_year"].isna().any()


def test_data_validation_report():
    assert VALIDATION_PATH.exists(), "Validation report missing. Run python -m src.data_validation"
    report = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    assert report["status"] == "PASSED"
    assert report["pass_rate_pct"] == 100.0
    assert report["dataset_rows"] >= 5000


def test_slice_metrics_and_fairness():
    assert SLICE_PATH.exists(), "Slice metrics missing. Run python -m src.train"
    report = json.loads(SLICE_PATH.read_text(encoding="utf-8"))
    assert report["total_slices_evaluated"] >= 10
    assert report["compliant_slices"] >= 10
    # Assert electric vehicles slice is evaluated
    ev_slices = [s for s in report["slices"] if s["slice_value"] == "Electric"]
    assert len(ev_slices) == 1
    assert ev_slices[0]["sla_status"] == "COMPLIANT"


def test_training_serving_feature_parity():
    """Verify engineer_features produces identical columns and derivations for single payload vs batch."""
    from src.data_prep import engineer_features

    single_payload = pd.DataFrame([{
        "year": 2023,
        "km_driven": 24000,
        "brand": "Tata",
        "car_model": "Nexon-EV",
        "fuel": "Electric",
        "seller_type": "Individual",
        "transmission": "Automatic",
        "owner": "First Owner",
    }])

    batch_payload = pd.concat([single_payload, single_payload], ignore_index=True)

    single_eng = engineer_features(single_payload)
    batch_eng = engineer_features(batch_payload)

    assert single_eng["vehicle_age"].iloc[0] == batch_eng["vehicle_age"].iloc[0]
    assert single_eng["km_per_year"].iloc[0] == batch_eng["km_per_year"].iloc[0]
    assert single_eng["is_luxury"].iloc[0] == batch_eng["is_luxury"].iloc[0]


def test_drift_psi_computation():
    """Verify PSI behaves correctly: identical distributions yield PSI ~0; shifted yields PSI > 0.25."""
    from src.drift import compute_psi

    np.random.seed(42)
    baseline = np.random.normal(50000, 10000, 1000)
    identical = np.random.normal(50000, 10000, 1000)
    shifted = np.random.normal(120000, 15000, 1000)

    psi_stable = compute_psi(baseline, identical)
    psi_drifted = compute_psi(baseline, shifted)

    assert psi_stable < 0.10, f"Expected stable PSI < 0.10, got {psi_stable}"
    assert psi_drifted > 0.25, f"Expected severe PSI > 0.25, got {psi_drifted}"


def test_model_performance_gates():
    if not METRICS_PATH.exists():
        pytest.skip("metrics.json missing — run python -m src.train")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    # Production MLOps Quality Gates
    assert metrics["r2"] >= 0.88, f"R2 score {metrics['r2']} failed quality gate (>= 0.88)"
    assert metrics["mae_inr"] <= 120_000, f"MAE {metrics['mae_inr']} exceeded SLA threshold (<= 120,000 INR)"
    assert metrics["accuracy"] >= 0.75, f"Accuracy {metrics['accuracy']} failed threshold (>= 0.75)"
    assert metrics["train_rows"] >= 5000, f"Training rows {metrics['train_rows']} below minimum required sample"


def test_governance_artifacts_exist():
    assert DRIFT_PATH.exists(), "Drift baseline artifact missing"
    assert FEATURE_IMP_PATH.exists(), "Feature importance artifact missing"
    assert CHAMPION_PATH.exists(), "Champion metadata artifact missing"

    feat_data = json.loads(FEATURE_IMP_PATH.read_text(encoding="utf-8"))
    assert len(feat_data) >= 5
    assert sum(f["importance_pct"] for f in feat_data) > 90.0
