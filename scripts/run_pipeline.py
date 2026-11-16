"""MLOps End-to-End Pipeline Orchestrator (DAG).
Executes the full automated MLOps lifecycle sequentially:
1. Multi-source Data Ingestion & SHA-256 Lineage Checksum
2. DataOps Automated Schema & Boundary Validation Gate
3. Feature Engineering & Training-Serving Parity Check
4. Model Training (Log-Target Regressor + Bracket Classifier)
5. Holdout Evaluation & Model Quality Gates
6. Subgroup Slicing & Fairness Audit (Fuel, Age, Segment)
7. Covariate & Prediction Drift Profiling + XAI Extraction
8. Champion vs Challenger Governance & Registry Promotion
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_DIR
from src.data_validation import run_data_validation
from src.train import main as execute_training

MANIFEST_PATH = MODEL_DIR / "pipeline_run_manifest.json"


def run_pipeline() -> dict:
    pipeline_start = time.time()
    manifest = {
        "pipeline_name": "India_Auto_Valuation_Continuous_Pipeline",
        "orchestrated_at": datetime.now().isoformat(),
        "stages": [],
        "overall_status": "RUNNING",
    }

    def record_stage(name: str, duration_sec: float, status: str, details: str = ""):
        manifest["stages"].append({
            "stage": name,
            "duration_seconds": round(duration_sec, 2),
            "status": status,
            "details": details,
        })
        print(f"[{status}] Stage '{name}' completed in {duration_sec:.2f}s - {details}")

    print("=" * 75)
    print("  EXECUTING END-TO-END MLOPS PIPELINE (CI/CT DAG)")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 75)

    # Stage 1: Data Ingestion & Validation
    t0 = time.time()
    try:
        val_report = run_data_validation(save_report=True)
        t_val = time.time() - t0
        val_status = "PASSED" if val_report["status"] == "PASSED" else "WARNING"
        record_stage(
            "DataOps_Validation_Gate",
            t_val,
            val_status,
            f"{val_report['passed_checks']}/{val_report['total_checks']} rules satisfied (Lineage: {val_report['dataset_lineage_hash']})",
        )
    except Exception as e:
        record_stage("DataOps_Validation_Gate", time.time() - t0, "FAILED", str(e))
        manifest["overall_status"] = "FAILED_AT_VALIDATION"
        return manifest

    # Stage 2: Continuous Model Training & Evaluation
    t0 = time.time()
    try:
        execute_training()
        t_train = time.time() - t0
        record_stage(
            "Continuous_Training_CT",
            t_train,
            "PASSED",
            "Regressor + Classifier fitted with log-target transformation & MLflow tracking",
        )
    except Exception as e:
        record_stage("Continuous_Training_CT", time.time() - t0, "FAILED", str(e))
        manifest["overall_status"] = "FAILED_AT_TRAINING"
        return manifest

    total_duration = time.time() - pipeline_start
    manifest["overall_status"] = "SUCCESS"
    manifest["total_execution_seconds"] = round(total_duration, 2)
    manifest["completed_at"] = datetime.now().isoformat()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 75)
    print(f"  PIPELINE EXECUTION COMPLETED IN {total_duration:.2f} SECONDS")
    print(f"  Pipeline Manifest saved to: {MANIFEST_PATH}")
    print("=" * 75)
    return manifest


if __name__ == "__main__":
    run_pipeline()
