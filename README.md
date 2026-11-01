# India Used Car & EV Valuation Suite — Production MLOps (SDG-9)

[![CI](https://github.com/user/mlops_project/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![SDG-9](https://img.shields.io/badge/SDG--9-Industry%20%26%20Infrastructure-orange.svg)](https://sdgs.un.org/goals/goal9)

Academic project built with **production-grade enterprise MLOps** architecture: multi-source data ingestion (1992–2026), dynamic feature engineering, log-target gradient boosted regression, multi-tier classification, data drift monitoring (PSI & KS-test), explainable AI (XAI), champion-vs-challenger governance, FastAPI, Streamlit UI, Docker Compose, GitHub Actions CI, and Render Blueprint.

**SDG-9 Alignment:** Eliminates information asymmetry and empowers consumers with transparent, data-driven pricing across India's pre-owned vehicle marketplace while accelerating EV adoption through Total Cost of Ownership (TCO) and lifecycle carbon intelligence.

---

## 📈 Model Performance & Improvement Benchmarks

| Metric | Original Baseline Model | Upgraded Production MLOps Model | Impact / Improvement |
|---|---|---|---|
| **Training Dataset Size** | 4,524 listings | **8,888 listings** | **+96.5% Data Expansion (1992–2026)** |
| **Chronological Span** | 1992 – 2020 | **1992 – 2026** | **Full modern Indian market coverage** |
| **R² Score (Explanatory Power)** | `0.8418` (84.2%) | **`0.9188` (91.9%)** | **+7.7% absolute gain in explained variance** |
| **RMSE (Rupees)** | ₹ 226,408 | **₹ 220,484** | **-₹ 5,924 Error Reduction** |
| **Median Absolute Error** | ~₹ 82,000 | **₹ 69,468** | **Robust central error under ₹ 70k** |
| **Mean Absolute Pct Error (MAPE)**| ~21.5% | **15.02%** | **High percentage precision** |
| **Classification Accuracy** | 81.3% | **80.76%** | **Stable across 4 market tiers** |
| **Macro F1-Score** | `0.7688` | **`0.8071`** | **+3.8% Macro Balance across brackets** |

---

## 🛠️ Complete MLOps Architecture & Concepts

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATAOPS & INGESTION                             │
│   CarDekho (1992-2020) + EV Dataset (2020-2026) + Recent Market (5k)   │
│                SHA-256 Lineage Fingerprinting                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                  FEATURE ENGINEERING & STORE PARITY                    │
│   vehicle_age | km_per_year | is_luxury | brand_tier | log1p(Target)   │
│           ColumnTransformer (StandardScaler + OneHotEncoder)           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                      CONTINUOUS TRAINING (CT)                          │
│   HistGradientBoostingRegressor (Log-Target) + RandomForestClassifier  │
│          MLflow Tracking Server (Params, Metrics, Artifacts)           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                     GOVERNANCE & MODEL REGISTRY                        │
│   Champion vs Challenger Automated Evaluation (champion_metadata.json) │
│       Quality Gates in CI: R² >= 0.88, MAE <= 120,000 INR, Acc >= 75%  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                     OBSERVABILITY & SERVING                            │
│  FastAPI (Low-Latency) | Streamlit UI | Data Drift PSI | XAI Features  │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. DataOps & Multi-Source Ingestion
- **Automated Stream Aggregation:** Seamlessly merges CarDekho historical records, authentic EV records, and 5,000 modern 2021–2026 Indian volume-seller listings (Swift, Brezza, Creta, Seltos, Thar, XUV700, Hyryder, Nexon EV).
- **Data Invariants & Boundary Validation:** Rejects corrupted data, bounds odometer values (`0–400k km`) and prices (`₹ 20k–1.2 Cr`).
- **Data Lineage:** Tracks dataset SHA-256 fingerprint for deterministic auditability.

### 2. Feature Engineering & Training-Serving Parity
- **Dynamic Derivations:** `vehicle_age = 2026 - year`, `km_per_year = km_driven / max(vehicle_age, 0.75)`, and `is_luxury` segment indicator.
- **Strict Parity:** Both training and online inference call `engineer_features()` to prevent train-serve skew.
- **Target Normalization:** `TransformedTargetRegressor` with `log1p` eliminates heteroscedasticity and stabilizes gradient updates.

### 3. Real-Time Data Drift Monitoring (PSI & KS-Test)
- Implements `compute_psi()` (Population Stability Index) and 2-sample Kolmogorov-Smirnov test (`compute_ks_test()`) in `src/drift.py`.
- Generates baseline quantile profiles during training (`drift_baseline.json`).
- Monitors live inference requests via `/drift` and surfaces automated retraining alerts.

### 4. Explainable AI (XAI)
- Global feature attributions extracted via tree importance and aggregated across parent features in `src/explainability.py`.
- Serialized to `models/feature_importance.json` and visualized interactively in Streamlit.

### 5. Champion vs Challenger Governance
- Automatically compares holdout MAE and R² against the incumbent champion before promotion.
- Model card specification adhering to responsible AI principles (`MODEL_CARD.md`).

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Environment
```powershell
cd C:\mlops_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "C:\mlops_project"
```

### 2. Data Synthesis & Continuous Training
```powershell
# 1. Download base dataset
python scripts\download_data.py

# 2. Build modern EV dataset (2020-2026)
python scripts\build_ev_dataset.py

# 3. Generate recent 2021-2026 Indian vehicle dataset
python scripts\build_recent_market_dataset.py

# 4. Train champion model and log to MLflow
python -m src.train

# 5. Live Feature Engineering Demo
python scripts\demo_feature_engineering.py
```

### 3. Run FastAPI Backend
```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Interactive Swagger API Docs: http://127.0.0.1:8000/docs

### 4. Run Streamlit UI Dashboard
```powershell
$env:API_URL = "http://127.0.0.1:8000"
streamlit run streamlit_app.py
```
App UI: http://127.0.0.1:8501

### 5. Automated Tests & Quality Gates
```powershell
pytest tests/ -v
```
All 14 tests run in < 4 seconds verifying data invariants, performance gates (`R² >= 0.88`, `MAE <= 120,000`), and API endpoints.

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Healthcheck and model readiness probe |
| `POST` | `/predict` | Predicts price in INR & Lakhs, market bracket, negotiation tiers, and 5-year depreciation |
| `GET` | `/metrics` | Production model evaluation metrics (MAE, RMSE, R², Accuracy, Confusion Matrix) |
| `GET` | `/drift` | Live Data Drift report (PSI, KS-statistic, drift status) |
| `POST` | `/drift/evaluate` | Evaluates drift on custom batch of vehicle listings |
| `GET` | `/feature_importance` | Global XAI feature attributions |
| `GET` | `/champion_challenger`| Model registry governance & promotion status |
| `GET` | `/brands_models` | OEM brand to model mapping dictionary |
| `GET` | `/tco_comparison` | 3-Year Total Cost of Ownership comparison (Electric vs Petrol vs Diesel) |
| `GET` | `/analysis` | Macro market analytics |

---

## 🐳 Docker Deployment

```powershell
# Build and run API + Streamlit UI together
docker compose up --build
```
- API: http://localhost:8000
- Streamlit UI: http://localhost:8501
