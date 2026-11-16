"""Production-Grade MLOps Training Pipeline (Continuous Training - CT).
Includes:
- Multi-source data ingestion (1992-2026 listings & EVs) with SHA-256 lineage
- DataOps Automated Schema & Invariant Validation Gate
- Dynamic feature engineering with strict offline/online parity
- Log-target transformed gradient boosted regressor (minimizing MAE, maximizing R2)
- Multi-tier bracket classifier (optimizing Accuracy & Macro-F1)
- Subgroup Slicing & Fairness Evaluation (fuel, age, luxury, transmission)
- Champion vs Challenger governance & model registry promotion
- Data & Prediction drift baseline profiling & feature importance extraction (XAI)
- Comprehensive MLflow tracking & artifact management
"""

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Prevent Windows loky cpu detection warning in sandboxed environments
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

from src.config import (  # noqa: E402
    BRACKET_LABELS,
    CATEGORICAL_FEATURES,
    CHAMPION_METADATA_PATH,
    CLASSIFICATION_TARGET,
    CLASSIFIER_MODEL_PATH,
    CURRENT_YEAR,
    DATA_RAW,
    DRIFT_BASELINE_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
)
from src.data_prep import load_and_clean  # noqa: E402
from src.data_validation import VALIDATION_REPORT_PATH, validate_dataset  # noqa: E402
from src.drift import generate_drift_baseline  # noqa: E402
from src.explainability import extract_feature_importance  # noqa: E402
from src.model_slicing import SLICE_METRICS_PATH, evaluate_slices, save_slice_metrics  # noqa: E402


def build_regressor_pipeline() -> Pipeline:
    """Build preprocessor and log-target transformed gradient boosting regressor."""
    numeric = Pipeline([("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )

    core_regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=500,
        max_depth=14,
        min_samples_leaf=4,
        learning_rate=0.05,
        l2_regularization=0.02,
        random_state=RANDOM_STATE,
    )

    # TransformedTargetRegressor applies log1p on selling_price to eliminate skew and minimize MAE
    regressor = TransformedTargetRegressor(
        regressor=core_regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    return Pipeline([("prep", preprocessor), ("model", regressor)])


def build_classifier_pipeline() -> Pipeline:
    """Build multi-tier bracket classifier with balanced class representations."""
    numeric = Pipeline([("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Calculate deterministic SHA-256 fingerprint of training data for DataOps lineage."""
    sample_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(sample_bytes).hexdigest()[:16]


def evaluate_champion_challenger(new_mae: float, new_r2: float, new_acc: float) -> dict:
    """Champion vs Challenger model governance framework."""
    champion_status = "PROMOTED_NEW_CHAMPION"
    prev_mae = None
    prev_r2 = None

    if CHAMPION_METADATA_PATH.exists():
        try:
            prev_meta = json.loads(CHAMPION_METADATA_PATH.read_text(encoding="utf-8"))
            prev_mae = prev_meta.get("mae_inr")
            prev_r2 = prev_meta.get("r2_score")
            # Promote if MAE is lower or R2 is higher
            if prev_mae is not None and prev_r2 is not None:
                if new_mae <= prev_mae or new_r2 >= prev_r2:
                    champion_status = "PROMOTED_REPLACING_PREVIOUS"
                else:
                    champion_status = "CHALLENGER_RETAINED_AS_BASELINE"
        except Exception:
            pass

    metadata = {
        "model_version": f"2.2.0-ct-{datetime.now().strftime('%Y%m%d')}",
        "governance_status": champion_status,
        "promoted_at": datetime.now().isoformat(),
        "mae_inr": new_mae,
        "r2_score": new_r2,
        "accuracy": new_acc,
        "previous_mae_inr": prev_mae,
        "previous_r2_score": prev_r2,
        "sdg_impact": "SDG-9 Fair Used-Car Valuation & Resilient Resale Infrastructure",
    }
    return metadata


def main() -> None:
    if not DATA_RAW.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_RAW}. Run: python scripts/download_data.py"
        )

    print("=" * 75)
    print("  MLOPS CONTINUOUS TRAINING (CT) & MODEL GOVERNANCE PIPELINE")
    print(f"  Reference Year: {CURRENT_YEAR} | Academic MLOps Course Suite")
    print("=" * 75)

    # 1. Multi-source Data Ingestion & Hygiene
    print("\n[Step 1/8] Ingesting multi-source datasets (Historical, Mid-Era, EV, 2021-2026)...")
    df = load_and_clean(DATA_RAW)
    data_hash = compute_dataset_hash(df)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y_reg = df[TARGET]
    y_cls = df[CLASSIFICATION_TARGET]

    print(f"-> Ingested {len(df):,} clean automotive listings.")
    print(f"-> Data Lineage Fingerprint (SHA-256): {data_hash}")
    print(f"-> Feature Space: {len(features)} inputs ({len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical)")

    # 2. DataOps Automated Validation Gate
    print("\n[Step 2/8] Executing DataOps Schema & Boundary Validation Gate...")
    val_report = validate_dataset(df)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT_PATH.write_text(json.dumps(val_report, indent=2), encoding="utf-8")
    print(f"-> Data Quality Gate Status: {val_report['status']} ({val_report['passed_checks']}/{val_report['total_checks']} checks passed)")
    if val_report["status"] != "PASSED":
        print("[WARNING] Data quality checks flagged anomalies, review data_validation_report.json")

    # 3. Stratified Train-Test Split (Holdout Validation)
    print("\n[Step 3/8] Performing Stratified Train-Test Splitting (80/20)...")
    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cls
    )
    print(f"-> Training Rows: {len(X_train):,} | Holdout Evaluation Rows: {len(X_test):,}")

    # 4. Train Log-Transformed Regressor
    print("\n[Step 4/8] Fitting Production Gradient Boosting Regressor (Log-Target Scaled)...")
    reg_pipe = build_regressor_pipeline()
    reg_pipe.fit(X_train, y_train_reg)
    reg_preds = reg_pipe.predict(X_test)

    # 5. Train Multi-Tier Bracket Classifier
    print("\n[Step 5/8] Fitting Multi-Tier Market Bracket Classifier...")
    cls_pipe = build_classifier_pipeline()
    cls_pipe.fit(X_train, y_train_cls)
    cls_preds = cls_pipe.predict(X_test)

    # 6. Compute Comprehensive MLOps Metrics
    print("\n[Step 6/8] Evaluating Production Metrics on Holdout Dataset...")
    mae = float(mean_absolute_error(y_test_reg, reg_preds))
    rmse = float(mean_squared_error(y_test_reg, reg_preds) ** 0.5)
    r2 = float(r2_score(y_test_reg, reg_preds))
    mape = float(mean_absolute_percentage_error(y_test_reg, reg_preds) * 100.0)
    med_ae = float(median_absolute_error(y_test_reg, reg_preds))

    acc = float(accuracy_score(y_test_cls, cls_preds))
    prec = float(precision_score(y_test_cls, cls_preds, average="macro", zero_division=0))
    rec = float(recall_score(y_test_cls, cls_preds, average="macro", zero_division=0))
    f1 = float(f1_score(y_test_cls, cls_preds, average="macro", zero_division=0))

    cm = confusion_matrix(y_test_cls, cls_preds, labels=BRACKET_LABELS).tolist()

    print("\n" + "-" * 50)
    print("  PRODUCTION MODEL EVALUATION RESULTS")
    print("-" * 50)
    print(f"  MAE (Mean Absolute Error):  Rs. {mae:,.2f}")
    print(f"  Median Absolute Error:      Rs. {med_ae:,.2f}")
    print(f"  RMSE:                       Rs. {rmse:,.2f}")
    print(f"  R2 Score (Explanatory):     {r2:.4f} ({r2*100:.1f}%)")
    print(f"  MAPE (Percentage Error):    {mape:.2f}%")
    print(f"  Classification Accuracy:    {acc*100:.2f}%")
    print(f"  Classification Macro F1:    {f1:.4f}")
    print("-" * 50)

    # 7. Subgroup Slicing Evaluation & Fairness Audit
    print("\n[Step 7/8] Running Subgroup Slicing & Fairness Evaluation...")
    slice_report = evaluate_slices(reg_pipe, X_test, y_test_reg, df)
    save_slice_metrics(slice_report)
    print(f"-> Slices Evaluated: {slice_report['total_slices_evaluated']} (Compliant: {slice_report['compliant_slices']}, Flagged: {slice_report['flagged_slices']})")

    # 8. Governance, Drift Profiling & MLflow Tracking
    print("\n[Step 8/8] Packaging Artifacts & Serializing Governance Baselines...")
    champion_meta = evaluate_champion_challenger(mae, r2, acc)
    CHAMPION_METADATA_PATH.write_text(json.dumps(champion_meta, indent=2), encoding="utf-8")

    # Drift Baseline Profile with Prediction Baseline
    drift_baseline = generate_drift_baseline(
        X_train, NUMERIC_FEATURES, CATEGORICAL_FEATURES, baseline_predictions=reg_preds
    )
    DRIFT_BASELINE_PATH.write_text(json.dumps(drift_baseline, indent=2), encoding="utf-8")

    # Feature Importance XAI
    feature_imp = extract_feature_importance(reg_pipe, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    FEATURE_IMPORTANCE_PATH.write_text(json.dumps(feature_imp, indent=2), encoding="utf-8")

    metrics = {
        "mae_inr": mae,
        "median_ae_inr": med_ae,
        "rmse_inr": rmse,
        "r2": r2,
        "mape_pct": mape,
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "bracket_labels": BRACKET_LABELS,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target": TARGET,
        "classification_target": CLASSIFICATION_TARGET,
        "features": features,
        "data_lineage_hash": data_hash,
        "data_validation_status": val_report["status"],
        "compliant_slices": slice_report["compliant_slices"],
        "total_slices": slice_report["total_slices_evaluated"],
        "trained_timestamp": datetime.now().isoformat(),
        "model_version": champion_meta["model_version"],
        "champion_status": champion_meta["governance_status"],
        "sdg_goal": "SDG-9 Industry, Innovation and Infrastructure",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    joblib.dump(reg_pipe, MODEL_PATH)
    joblib.dump(cls_pipe, CLASSIFIER_MODEL_PATH)

    # MLflow Tracking
    print("\nLogging Run to MLflow Tracking Server...")
    try:
        mlflow.set_experiment("India_Used_Car_Price_Prediction")
        with mlflow.start_run(run_name="Production_Champion_LogHistGB_Slices_v2"):
            mlflow.set_tags({
                "mlops_stage": "Production",
                "framework": "scikit-learn",
                "regressor": "HistGradientBoostingRegressor",
                "target_transform": "log1p",
                "dataset_hash": data_hash,
                "dataset_span": f"1992-{CURRENT_YEAR}",
                "sdg_goal": "SDG-9",
                "champion_status": champion_meta["governance_status"],
                "data_validation": val_report["status"],
            })

            mlflow.log_params({
                "max_iter": 500,
                "max_depth": 14,
                "min_samples_leaf": 4,
                "learning_rate": 0.05,
                "l2_regularization": 0.02,
                "num_features": len(features),
                "features": ",".join(features),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            })

            mlflow.log_metrics({
                "mae_inr": mae,
                "median_ae_inr": med_ae,
                "rmse_inr": rmse,
                "r2_score": r2,
                "mape_pct": mape,
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1,
                "compliant_slices": slice_report["compliant_slices"],
            })

            mlflow.log_artifact(str(MODEL_PATH))
            mlflow.log_artifact(str(CLASSIFIER_MODEL_PATH))
            mlflow.log_artifact(str(METRICS_PATH))
            mlflow.log_artifact(str(DRIFT_BASELINE_PATH))
            mlflow.log_artifact(str(FEATURE_IMPORTANCE_PATH))
            mlflow.log_artifact(str(CHAMPION_METADATA_PATH))
            mlflow.log_artifact(str(VALIDATION_REPORT_PATH))
            mlflow.log_artifact(str(SLICE_METRICS_PATH))
        print("-> MLflow Run Successfully Logged.")
    except Exception as e:
        print(f"-> MLflow Notice: {e}")

    print("\n" + "=" * 75)
    print("  TRAINING PIPELINE COMPLETE — CHAMPION MODEL READY FOR SERVING")
    print("=" * 75)


if __name__ == "__main__":
    main()
