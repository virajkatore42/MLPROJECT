"""MLOps Data & Concept Drift Observability Suite.
Implements:
- Covariate Shift (Feature Drift): Population Stability Index (PSI) & 2-Sample KS-Test
- Concept / Output Drift: Prediction Distribution PSI vs Baseline Predictions
- Automated Drift Threshold Alerting & Continuous Retraining Recommendation
- Synthetic Drift Traffic Simulator for testing & live demonstration
"""

from typing import Any
import numpy as np
import pandas as pd


def compute_psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Compute Population Stability Index (PSI) between baseline and live distributions."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Quantile bin edges based on expected baseline
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) <= 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(max(0.0, psi_val))


def compute_ks_test(expected: np.ndarray | pd.Series, actual: np.ndarray | pd.Series) -> dict[str, float]:
    """Compute 2-sample Kolmogorov-Smirnov statistic for continuous distributions."""
    try:
        from scipy.stats import ks_2samp
        stat, p_val = ks_2samp(expected, actual)
        return {"statistic": float(stat), "p_value": float(p_val)}
    except Exception:
        e_sorted = np.sort(expected)
        a_sorted = np.sort(actual)
        all_vals = np.concatenate([e_sorted, a_sorted])
        e_cdf = np.searchsorted(e_sorted, all_vals, side="right") / len(e_sorted)
        a_cdf = np.searchsorted(a_sorted, all_vals, side="right") / len(a_sorted)
        stat = float(np.max(np.abs(e_cdf - a_cdf)))
        return {"statistic": stat, "p_value": 0.05 if stat < 0.2 else 0.001}


def generate_drift_baseline(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    baseline_predictions: np.ndarray | None = None,
) -> dict[str, Any]:
    """Generate and serialize reference statistical profiles from training data."""
    baseline: dict[str, Any] = {
        "record_count": int(len(df)),
        "numeric_profiles": {},
        "categorical_profiles": {},
        "prediction_profile": None,
    }

    for col in numeric_features:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            baseline["numeric_profiles"][col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "p25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "p75": float(series.quantile(0.75)),
                "max": float(series.max()),
                "sample_values": series.sample(min(1000, len(series)), random_state=42).tolist(),
            }

    for col in categorical_features:
        if col in df.columns:
            counts = df[col].value_counts(normalize=True).to_dict()
            baseline["categorical_profiles"][col] = {str(k): float(v) for k, v in counts.items()}

    if baseline_predictions is not None and len(baseline_predictions) > 0:
        pred_series = pd.Series(baseline_predictions).dropna()
        baseline["prediction_profile"] = {
            "mean": float(pred_series.mean()),
            "std": float(pred_series.std()),
            "median": float(pred_series.median()),
            "sample_values": pred_series.sample(min(1000, len(pred_series)), random_state=42).tolist(),
        }

    return baseline


def evaluate_drift(
    live_df: pd.DataFrame,
    baseline_dict: dict[str, Any],
    live_predictions: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate drift between incoming live samples and baseline profile."""
    report: dict[str, Any] = {
        "live_samples": int(len(live_df)),
        "feature_metrics": {},
        "overall_status": "STABLE",
        "max_psi": 0.0,
        "average_psi": 0.0,
        "drift_detected": False,
        "retraining_recommended": False,
        "prediction_drift": None,
    }

    psi_scores = []
    numeric_profiles = baseline_dict.get("numeric_profiles", {})

    for col, profile in numeric_profiles.items():
        if col in live_df.columns:
            baseline_vals = np.array(profile["sample_values"])
            live_vals = pd.to_numeric(live_df[col], errors="coerce").dropna().to_numpy()

            if len(live_vals) >= 5:
                psi = compute_psi(baseline_vals, live_vals)
                ks = compute_ks_test(baseline_vals, live_vals)
                psi_scores.append(psi)

                if psi < 0.1:
                    status = "STABLE"
                elif psi < 0.25:
                    status = "MODERATE_DRIFT"
                else:
                    status = "SIGNIFICANT_DRIFT"

                report["feature_metrics"][col] = {
                    "psi": round(psi, 4),
                    "ks_statistic": round(ks["statistic"], 4),
                    "status": status,
                    "live_mean": round(float(np.mean(live_vals)), 2),
                    "baseline_mean": round(profile["mean"], 2),
                }

    # Evaluate prediction / concept drift if predictions are provided
    pred_profile = baseline_dict.get("prediction_profile")
    if pred_profile and live_predictions is not None and len(live_predictions) >= 5:
        base_preds = np.array(pred_profile["sample_values"])
        live_preds = np.array(live_predictions)
        pred_psi = compute_psi(base_preds, live_preds)
        pred_ks = compute_ks_test(base_preds, live_preds)
        psi_scores.append(pred_psi)

        report["prediction_drift"] = {
            "psi": round(pred_psi, 4),
            "ks_statistic": round(pred_ks["statistic"], 4),
            "status": "SIGNIFICANT_DRIFT" if pred_psi >= 0.25 else ("MODERATE_DRIFT" if pred_psi >= 0.10 else "STABLE"),
            "live_mean_prediction": round(float(np.mean(live_preds)), 2),
            "baseline_mean_prediction": round(pred_profile["mean"], 2),
        }

    if psi_scores:
        max_psi = float(np.max(psi_scores))
        avg_psi = float(np.mean(psi_scores))
        report["max_psi"] = round(max_psi, 4)
        report["average_psi"] = round(avg_psi, 4)

        if max_psi >= 0.25:
            report["overall_status"] = "DRIFT_DETECTED"
            report["drift_detected"] = True
            report["retraining_recommended"] = True
        elif max_psi >= 0.10:
            report["overall_status"] = "MODERATE_DRIFT"
            report["drift_detected"] = False
            report["retraining_recommended"] = False
        else:
            report["overall_status"] = "STABLE"
            report["drift_detected"] = False
            report["retraining_recommended"] = False

    return report


def simulate_drift_dataset(scenario: str = "high_mileage_aging") -> pd.DataFrame:
    """Generates synthetic incoming batches representing real-world distribution shifts."""
    if scenario == "high_mileage_aging":
        # Severe covariate shift: Older cars with very high mileage
        return pd.DataFrame([
            {"year": 2008, "km_driven": 185000, "brand": "Maruti", "car_model": "Swift", "fuel": "Diesel", "seller_type": "Individual", "transmission": "Manual", "owner": "Third Owner"},
            {"year": 2007, "km_driven": 210000, "brand": "Hyundai", "car_model": "Santro", "fuel": "Petrol", "seller_type": "Individual", "transmission": "Manual", "owner": "Fourth & Above Owner"},
            {"year": 2010, "km_driven": 165000, "brand": "Tata", "car_model": "Indica", "fuel": "Diesel", "seller_type": "Individual", "transmission": "Manual", "owner": "Second Owner"},
            {"year": 2009, "km_driven": 195000, "brand": "Mahindra", "car_model": "Bolero", "fuel": "Diesel", "seller_type": "Individual", "transmission": "Manual", "owner": "Third Owner"},
            {"year": 2006, "km_driven": 240000, "brand": "Toyota", "car_model": "Innova", "fuel": "Diesel", "seller_type": "Individual", "transmission": "Manual", "owner": "Second Owner"},
            {"year": 2008, "km_driven": 178000, "brand": "Honda", "car_model": "City", "fuel": "Petrol", "seller_type": "Individual", "transmission": "Manual", "owner": "Third Owner"},
        ])
    elif scenario == "luxury_surge":
        # Covariate & Concept shift: Sudden influx of luxury sports & executive cars
        return pd.DataFrame([
            {"year": 2024, "km_driven": 8000, "brand": "BMW", "car_model": "3-Series", "fuel": "Petrol", "seller_type": "Dealer", "transmission": "Automatic", "owner": "First Owner"},
            {"year": 2023, "km_driven": 14000, "brand": "Mercedes-Benz", "car_model": "C-Class", "fuel": "Diesel", "seller_type": "Dealer", "transmission": "Automatic", "owner": "First Owner"},
            {"year": 2025, "km_driven": 5000, "brand": "Audi", "car_model": "A4", "fuel": "Petrol", "seller_type": "Dealer", "transmission": "Automatic", "owner": "First Owner"},
            {"year": 2024, "km_driven": 11000, "brand": "Volvo", "car_model": "XC60", "fuel": "Hybrid", "seller_type": "Dealer", "transmission": "Automatic", "owner": "First Owner"},
            {"year": 2023, "km_driven": 16000, "brand": "Porsche", "car_model": "Macan", "fuel": "Petrol", "seller_type": "Dealer", "transmission": "Automatic", "owner": "First Owner"},
        ])
    else:
        # In-distribution stable reference traffic
        return pd.DataFrame([
            {"year": 2022, "km_driven": 35000, "brand": "Maruti", "car_model": "Swift", "fuel": "Petrol", "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner"},
            {"year": 2023, "km_driven": 22000, "brand": "Hyundai", "car_model": "Creta", "fuel": "Diesel", "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner"},
            {"year": 2021, "km_driven": 45000, "brand": "Tata", "car_model": "Nexon", "fuel": "Petrol", "seller_type": "Dealer", "transmission": "Manual", "owner": "First Owner"},
            {"year": 2024, "km_driven": 15000, "brand": "Kia", "car_model": "Seltos", "fuel": "Petrol", "seller_type": "Individual", "transmission": "Automatic", "owner": "First Owner"},
            {"year": 2022, "km_driven": 38000, "brand": "Mahindra", "car_model": "XUV700", "fuel": "Diesel", "seller_type": "Individual", "transmission": "Automatic", "owner": "First Owner"},
        ])
