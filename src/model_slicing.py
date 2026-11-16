"""MLOps Model Slicing & Subgroup Evaluation Suite.
Evaluates model performance across critical data slices to detect:
- Algorithmic bias and disparity across subgroups
- Underrepresented segment performance (e.g. EVs vs ICE)
- Subgroup SLA violations (e.g. older vintage vehicles vs new cars)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.config import MODEL_DIR

SLICE_METRICS_PATH = MODEL_DIR / "slice_metrics.json"


def evaluate_slices(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    original_df: pd.DataFrame,
) -> dict[str, Any]:
    """Computes slice-level metrics across fuel, age, brand segment, and transmission."""
    eval_df = X_test.copy()
    eval_df["y_true"] = y_test.values
    eval_df["y_pred"] = model.predict(X_test)
    eval_df["abs_error"] = np.abs(eval_df["y_true"] - eval_df["y_pred"])
    eval_df["pct_error"] = (eval_df["abs_error"] / eval_df["y_true"]) * 100.0

    total_test_rows = len(eval_df)

    # Derived slicing categories
    if "vehicle_age" in eval_df.columns:
        eval_df["age_bracket"] = pd.cut(
            eval_df["vehicle_age"],
            bins=[-1, 3, 7, 100],
            labels=["New (0-3 yrs)", "Mid-Age (4-7 yrs)", "Older (8+ yrs)"],
        )
    else:
        eval_df["age_bracket"] = "Unknown"

    if "is_luxury" in eval_df.columns:
        eval_df["market_segment"] = eval_df["is_luxury"].map(
            {1: "Luxury OEM", 0: "Mass-Market OEM"}
        )
    else:
        eval_df["market_segment"] = "Standard"

    slice_definitions = [
        {"dimension": "Fuel / Powertrain", "column": "fuel"},
        {"dimension": "Vehicle Age Bracket", "column": "age_bracket"},
        {"dimension": "Market Tier", "column": "market_segment"},
        {"dimension": "Transmission Type", "column": "transmission"},
        {"dimension": "Ownership History", "column": "owner"},
    ]

    all_slices = []

    for s_def in slice_definitions:
        dim = s_def["dimension"]
        col = s_def["column"]
        if col not in eval_df.columns:
            continue

        grouped = eval_df.groupby(col, observed=True)
        for val, group in grouped:
            if len(group) == 0:
                continue

            y_t = group["y_true"].values
            y_p = group["y_pred"].values

            mae = float(mean_absolute_error(y_t, y_p))
            rmse = float(mean_squared_error(y_t, y_p) ** 0.5)
            mape = float(mean_absolute_percentage_error(y_t, y_p) * 100.0)
            
            # R2 requires at least 2 distinct values
            if len(np.unique(y_t)) > 1:
                r2_val = float(r2_score(y_t, y_p))
            else:
                r2_val = 0.0

            # SLA Rule: Flag slice if MAPE > 35% or MAE > 250,000 INR
            is_compliant = (mape <= 35.0) and (mae <= 250_000)

            all_slices.append({
                "dimension": dim,
                "slice_value": str(val),
                "sample_count": int(len(group)),
                "representation_pct": round((len(group) / total_test_rows) * 100.0, 1),
                "mae_inr": round(mae, 0),
                "rmse_inr": round(rmse, 0),
                "mape_pct": round(mape, 2),
                "r2_score": round(r2_val, 4),
                "sla_status": "COMPLIANT" if is_compliant else "FLAGGED_HIGH_VARIANCE",
            })

    # Summary analysis
    flagged = [s for s in all_slices if s["sla_status"] != "COMPLIANT"]

    report = {
        "evaluated_at": datetime.now().isoformat(),
        "total_test_samples": total_test_rows,
        "total_slices_evaluated": len(all_slices),
        "compliant_slices": len(all_slices) - len(flagged),
        "flagged_slices": len(flagged),
        "slices": all_slices,
    }
    return report


def save_slice_metrics(report: dict[str, Any]) -> None:
    """Serialize slice metrics to models directory."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SLICE_METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
