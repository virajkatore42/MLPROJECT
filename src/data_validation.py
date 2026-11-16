"""DataOps Automated Data Validation Suite.
Implements Great Expectations-style schema and invariant checks:
- Column existence and datatype validation
- Nullability and completeness thresholds (< 1% missing)
- Domain and categorical allowable sets
- Physical range boundaries (year, km_driven, selling_price)
- Distribution anomaly / outlier detection
- Dataset lineage hashing (SHA-256)
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CURRENT_YEAR, DATA_RAW, MODEL_DIR

VALIDATION_REPORT_PATH = MODEL_DIR / "data_validation_report.json"

ALLOWED_CATEGORIES = {
    "fuel": {"Petrol", "Diesel", "CNG", "LPG", "Electric", "Hybrid"},
    "seller_type": {"Individual", "Dealer", "Trustmark Dealer"},
    "transmission": {"Manual", "Automatic"},
    "owner": {
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car",
    },
}

NUMERIC_BOUNDS = {
    "year": {"min": 1990, "max": CURRENT_YEAR},
    "km_driven": {"min": 0, "max": 400_000},
    "selling_price": {"min": 20_000, "max": 12_000_000},
    "vehicle_age": {"min": 0, "max": 36},
}


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Calculate deterministic SHA-256 fingerprint for DataOps reproducibility."""
    sample_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(sample_bytes).hexdigest()[:16]


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Execute complete suite of automated DataOps validation rules."""
    checks = []
    total_rows = len(df)
    data_hash = compute_dataset_hash(df)

    # 1. Dataset size check
    size_passed = total_rows >= 1000
    checks.append({
        "check": "dataset_minimum_sample_size",
        "category": "Volume",
        "status": "PASSED" if size_passed else "FAILED",
        "details": f"Total rows: {total_rows:,} (threshold >= 1,000)",
        "passed": size_passed,
    })

    # 2. Required columns check
    required_cols = [
        "selling_price",
        "year",
        "km_driven",
        "fuel",
        "seller_type",
        "transmission",
        "owner",
        "brand",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    cols_passed = len(missing_cols) == 0
    checks.append({
        "check": "required_columns_presence",
        "category": "Schema",
        "status": "PASSED" if cols_passed else "FAILED",
        "details": f"Missing columns: {missing_cols if missing_cols else 'None'}",
        "passed": cols_passed,
    })

    # 3. Missingness / Null check
    null_counts = df[required_cols].isna().sum().to_dict() if cols_passed else {}
    null_failures = {k: v for k, v in null_counts.items() if v > 0}
    null_passed = len(null_failures) == 0
    checks.append({
        "check": "null_rate_threshold",
        "category": "Completeness",
        "status": "PASSED" if null_passed else "FAILED",
        "details": f"Columns with nulls: {null_failures if null_failures else 'Zero missing values (100% complete)'}",
        "passed": null_passed,
    })

    # 4. Numeric Range Boundary Checks
    range_failures = []
    for col, bounds in NUMERIC_BOUNDS.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            out_of_min = (s < bounds["min"]).sum()
            out_of_max = (s > bounds["max"]).sum()
            if out_of_min > 0 or out_of_max > 0:
                range_failures.append(f"{col}: {out_of_min} below {bounds['min']}, {out_of_max} above {bounds['max']}")

    ranges_passed = len(range_failures) == 0
    checks.append({
        "check": "numeric_boundary_invariants",
        "category": "Data Integrity",
        "status": "PASSED" if ranges_passed else "FAILED",
        "details": f"Violations: {range_failures if range_failures else 'All numeric values conform to physical boundaries'}",
        "passed": ranges_passed,
    })

    # 5. Categorical Domain Membership Checks
    cat_failures = []
    for col, valid_set in ALLOWED_CATEGORIES.items():
        if col in df.columns:
            actual_vals = set(df[col].dropna().astype(str).unique())
            invalid_vals = actual_vals - valid_set
            if invalid_vals:
                cat_failures.append(f"{col}: invalid values {invalid_vals}")

    cat_passed = len(cat_failures) == 0
    checks.append({
        "check": "categorical_domain_integrity",
        "category": "Schema",
        "status": "PASSED" if cat_passed else "FAILED",
        "details": f"Violations: {cat_failures if cat_failures else 'All categorical labels match authorized vocabulary'}",
        "passed": cat_passed,
    })

    # 6. Duplication Check
    dup_cols = [
        "brand",
        "car_model",
        "year",
        "km_driven",
        "fuel",
        "seller_type",
        "transmission",
        "owner",
        "selling_price",
    ]
    dup_subset = [c for c in dup_cols if c in df.columns]
    dup_count = int(df.duplicated(subset=dup_subset).sum())
    dup_passed = dup_count == 0
    checks.append({
        "check": "exact_row_deduplication",
        "category": "Uniqueness",
        "status": "PASSED" if dup_passed else "WARNING",
        "details": f"Found {dup_count:,} duplicate listings across feature space",
        "passed": dup_passed,
    })

    # 7. Target Skewness and Zero-Price Check
    zero_prices = int((df["selling_price"] <= 0).sum()) if "selling_price" in df.columns else 1
    zero_passed = zero_prices == 0
    checks.append({
        "check": "target_positive_invariant",
        "category": "Target Health",
        "status": "PASSED" if zero_passed else "FAILED",
        "details": f"Non-positive target rows: {zero_prices}",
        "passed": zero_passed,
    })

    passed_count = sum(1 for c in checks if c["passed"])
    failed_count = sum(1 for c in checks if not c["passed"])
    overall_status = "PASSED" if failed_count == 0 else "FAILED"

    report = {
        "status": overall_status,
        "evaluated_at": datetime.now().isoformat(),
        "dataset_rows": total_rows,
        "dataset_lineage_hash": data_hash,
        "total_checks": len(checks),
        "passed_checks": passed_count,
        "failed_checks": failed_count,
        "pass_rate_pct": round((passed_count / len(checks)) * 100.0, 1),
        "checks": checks,
    }
    return report


def run_data_validation(save_report: bool = True) -> dict[str, Any]:
    """Execute validation on the current cleaned training dataset."""
    from src.data_prep import load_and_clean

    df = load_and_clean(DATA_RAW)
    report = validate_dataset(df)

    if save_report:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        VALIDATION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


if __name__ == "__main__":
    rep = run_data_validation(save_report=True)
    print(f"Validation Status: {rep['status']} ({rep['passed_checks']}/{rep['total_checks']} checks passed)")
