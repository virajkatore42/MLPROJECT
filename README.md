# India Pre-Owned Vehicle Valuation Suite — Production MLOps Course Suite

[![CI](https://github.com/user/mlops_project/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![SDG-9](https://img.shields.io/badge/SDG--9-Industry%20%26%20Infrastructure-orange.svg)](https://sdgs.un.org/goals/goal9)

Production-grade Machine Learning Operations (MLOps) architecture engineered for the academic curriculum: multi-source data ingestion with SHA-256 lineage, automated DataOps schema validation gate, strict training-serving feature parity, log-target gradient boosted regression (`R² = 0.9209`), subgroup slicing and model fairness evaluation, real-time covariate and prediction drift monitoring (PSI & KS-test), operational serving telemetry with P95 latency tracking, champion-challenger registry governance, and CI/CD quality gates.

---

## 📈 Model Performance & Production Benchmarks

| Metric | Original Baseline Model | Production MLOps Champion | Impact / Improvement |
|---|---|---|---|
| **Training Dataset Size** | 4,524 listings | **12,888 listings** | **+184.8% Massive Data Expansion (1992–2026)** |
| **Chronological Span** | 1992 – 2020 | **1992 – 2026** | **Full modern Indian market coverage** |
| **Data Quality Gate** | Manual checks | **Automated (7/7 Rules)** | **100% Pass Rate via Great Expectations style gate** |
| **R² Score (Explanatory Power)** | `0.8418` (84.2%) | **`0.9522` (95.2%)** | **+11.0% absolute gain in explained variance** |
| **RMSE (Rupees)** | ₹ 226,408 | **₹ 184,843** | **-₹ 41,565 Substantial Error Drop** |
| **Median Absolute Error** | ~₹ 82,000 | **₹ 80,188** | **Robust central error across diverse tiers** |
| **Mean Absolute Pct Error (MAPE)**| ~21.5% | **14.19%** | **High percentage precision** |
| **Subgroup Slices Evaluated** | 0 (Unmonitored) | **18 Slices across Fuel & Age**| **17/18 Slices Compliant with SLA** |
| **Classification Accuracy** | 81.3% | **81.73%** | **Stable across 4 market tiers** |
| **Macro F1-Score** | `0.7688` | **`0.8202`** | **+5.1% Macro Balance across brackets** |
| **Inference Latency SLA** | Unprofiled | **P50: ~1.2 ms \| P95: ~3.8 ms**| **Ultra low-latency production serving** |

---

## 🛠️ Complete MLOps Architecture & Core Concepts

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATAOPS & INGESTION                             │
│   CarDekho (1992-2020) + EV Dataset (2020-2026) + Recent Market (9k)   │
│         SHA-256 Lineage Hash: d56ce3b668c03014                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                  DATA QUALITY & INVARIANT GATE                         │
│   Automated Schema & Range Assertions (src/data_validation.py)         │
│   Pass Rate: 100% (7/7 Rules) -> data_validation_report.json           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                  FEATURE STORE & TRAINING-SERVING PARITY               │
│   Dynamic Derivations: vehicle_age, km_per_year, is_luxury             │
│   Preprocessed via ColumnTransformer (StandardScaler + OneHotEncoder)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                      CONTINUOUS TRAINING (CT) DAG                      │
│   HistGradientBoostingRegressor (Log-Target) + RandomForestClassifier  │
│   End-to-End Orchestrator: scripts/run_pipeline.py                     │
│   MLflow Experiment Tracking Server (Params, Metrics, 8 Artifacts)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                   SUBGROUP SLICING & FAIRNESS AUDIT                    │
│   Evaluates 18 Subgroups across Fuel, Vehicle Age, Market Tier         │
│   Flags Disparities & Generates slice_metrics.json                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                     GOVERNANCE & MODEL REGISTRY                        │
│   Champion vs Challenger Automated Evaluation (champion_metadata.json) │
│   Quality Gates in CI: R² >= 0.88, MAE <= 120,000 INR, Acc >= 75%      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                     OBSERVABILITY & SERVING                            │
│   FastAPI Model Gateway (Low-Latency REST + Batch Vectorized Endpoint) │
│   Serving Telemetry: P50/P95/P99 Latency Profiler & Audit Logger       │
│   Covariate Drift (Feature PSI/KS) + Prediction / Concept Drift        │
│   Automated Retraining Alerting when PSI >= 0.25                       │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. DataOps & Schema Validation Gate
- **Automated Stream Aggregation:** Merges historical CarDekho, authentic electric vehicle records, and modern 2021–2026 volume sellers (8,888 records).
- **Automated Validation Invariants (`src/data_validation.py`):** Great Expectations-style automated assertion checks for null rate thresholds (<1%), physical boundaries (`year ∈ [1990, 2026]`, `km ∈ [0, 400k]`, `price ∈ [20k, 12M]`), categorical vocabulary membership, and exact row deduplication.
- **Cryptographic Lineage:** Tracks dataset SHA-256 fingerprint (`5babb492e291db9e`) for auditability and reproducibility.

### 2. Feature Engineering & Strict Training-Serving Parity
- **Dynamic Derivations:** `vehicle_age = 2026 - year`, `km_per_year = km_driven / max(vehicle_age, 0.75)`, and `is_luxury` segment indicator.
- **Zero Train-Serve Skew:** Both offline training and online serving invoke the identical `engineer_features()` pipeline.
- **Logarithmic Target Normalization:** `TransformedTargetRegressor` with `log1p` eliminates heteroscedasticity, minimizing skew and percentage errors.

### 3. Model Slicing & Fairness Evaluation
- **Subgroup Breakdown (`src/model_slicing.py`):** Evaluates model error across 18 distinct market slices (Fuel: Electric, Hybrid, Diesel, Petrol, CNG; Age: New [0-3 yrs], Mid-Age [4-7 yrs], Older [8+ yrs]; Tier: Mass-Market vs Luxury).
- **Disparity Detection:** Verifies that no slice experiences catastrophic failure or unfair bias (e.g., Electric vehicles achieve an outstanding `R² = 0.9589` and `MAPE = 8.82%`).

### 4. Real-Time Observability & Drift Monitoring
- **Covariate Shift Monitoring:** Implements Population Stability Index (PSI) and 2-sample Kolmogorov-Smirnov test (`compute_ks_test()`) on incoming features against training baseline.
- **Concept / Output Drift Monitoring:** Tracks the distribution of model predictions (`y_hat`) over time compared to training baseline predictions.
- **Automated Retraining Alert:** Automatically flags `retraining_recommended: True` when max PSI breaches safety threshold (`>= 0.25`).
- **Interactive Drift Simulator:** Built-in UI simulator allowing the evaluation of aging fleet shifts and luxury market surges.

### 5. Serving Telemetry & Operational Latency SLA
- **Production Telemetry (`src/telemetry.py`):** Real-time tracking of request counts, P50 latency, P95 latency, P99 latency, and average price.
- **Persistent Prediction Audit:** Every prediction is logged to `data/telemetry/prediction_audit.jsonl` with request ID, timestamp, inputs, outputs, and execution latency.
- **High-Throughput Batch Endpoint (`/predict/batch`):** Vectorized batch prediction for fleet appraisals with aggregated latency metrics.

### 6. Champion vs Challenger Model Governance
- **Automated Promotion Gate:** Challenger models are automatically compared against the incumbent champion on holdout MAE and R² before promotion.
- **Model Card:** Comprehensive technical documentation in [`MODEL_CARD.md`](MODEL_CARD.md).
- **Continuous Training (CT) Pipeline Orchestrator:** `scripts/run_pipeline.py` executes the entire DAG with execution manifest (`pipeline_run_manifest.json`).

---

## 🚀 Quickstart & Setup

### 1. Environment Setup
```powershell
cd C:\mlops_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "C:\mlops_project"
```

### 2. End-to-End MLOps Pipeline Execution
```powershell
# Executes complete DAG: Data Ingestion -> Validation Gate -> CT -> Slicing -> Drift Profiling -> Promotion
python scripts/run_pipeline.py
```

### 3. Run Automated Test Suite & Quality Gates
```powershell
pytest tests/ -v
```
All 21 unit & integration tests run in < 8 seconds verifying data invariants, training-serving parity, latency SLAs, model performance gates (`R² >= 0.88`, `MAE <= 120k`), and REST endpoints.

### 4. Start FastAPI Production Gateway
```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- Interactive Swagger API Docs: http://127.0.0.1:8000/docs
- Live Health Probe: http://127.0.0.1:8000/health
- Serving Telemetry: http://127.0.0.1:8000/telemetry

### 5. Start Streamlit MLOps Dashboard
```powershell
$env:API_URL = "http://127.0.0.1:8000"
streamlit run streamlit_app.py
```
App UI: http://127.0.0.1:8501

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Healthcheck, model readiness probe & active version |
| `POST` | `/predict` | Single vehicle prediction with latency profiling & audit logging |
| `POST` | `/predict/batch` | High-throughput vectorized batch prediction with batch latency profile |
| `GET` | `/telemetry` | Real-time serving telemetry (P50/P95 latency percentiles, throughput) |
| `GET` | `/data/validation` | DataOps schema & invariant validation report (7 rules, 100% pass) |
| `GET` | `/metrics/slices` | Subgroup slicing & fairness evaluation across 18 market slices |
| `GET` | `/metrics` | Production model evaluation metrics (MAE, RMSE, R², Accuracy, Confusion Matrix) |
| `GET` | `/drift` | Live Covariate & Concept Drift report (PSI, KS-statistic, alert status) |
| `POST` | `/drift/evaluate` | Evaluates drift on custom batch of incoming vehicle listings |
| `GET` | `/feature_importance` | Global XAI feature attributions |
| `GET` | `/champion_challenger`| Model registry governance & promotion status |
| `GET` | `/brands_models` | OEM brand to model mapping dictionary |

---

## 🐳 Docker Deployment

```powershell
# Build and run API + Streamlit UI together
docker compose up --build
```
- API Gateway: http://localhost:8000
- Streamlit MLOps Dashboard: http://localhost:8501
