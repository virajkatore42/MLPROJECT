# Model Card: India Pre-Owned Vehicle Valuation MLOps Suite (SDG-9)

## 1. Model Details
- **Model Name:** AutoValuate AI Resale Price Regressor & Bracket Classifier
- **Model Version:** `2.2.0-ct-enterprise`
- **Model Architecture:** Dual-Model Production Ensemble:
  1. **Continuous Regressor:** `TransformedTargetRegressor(HistGradientBoostingRegressor)` with dynamic logarithmic target scaling (`func=np.log1p, inverse_func=np.expm1`), `max_iter=500`, `max_depth=14`, `l2_regularization=0.02`.
  2. **Tier Classifier:** `RandomForestClassifier` predicting 4 discretized market price brackets (`Budget`, `Mid-Range`, `Premium`, `Luxury`).
- **Release Date:** November 2026
- **License:** MIT / Academic MLOps Production Course Suite

---

## 2. Intended Use & Domain
- **Primary Use Case:** Transparent, data-driven pre-owned passenger vehicle valuation across conventional and electric vehicles (EVs) in the Indian automotive market.
- **Target Audience:** Consumer valuation engines, automotive dealerships, insurance risk auditors, digital lending platforms.
- **Out-of-Scope Uses:** Heavy commercial trucks/buses, salvage/total-loss auction write-offs, exotic supercars (> ₹1.2 Crore).

---

## 3. DataOps & Schema Validation Gate
The dataset integrates multi-stream data ingestion across **1992 through 2026**:
1. **Historical CarDekho Dataset:** 4,340 historical listings (1992–2020).
2. **Indian Electric Vehicle (EV) Dataset:** 1,000 authentic and augmented EV listings (Tata Nexon-EV, Tiago-EV, MG ZS-EV, Mahindra XUV400, Hyundai Ioniq 5, Kia EV6).
3. **Modern 2021–2026 Market Dataset:** 5,000 modern listings covering high-volume market leaders (Swift, Brezza, Creta, Seltos, Thar, XUV700, Hyryder).
- **Total Ingested Listings:** 12,888 verified records.
- **Lineage Fingerprint:** Cryptographic SHA-256 hash `d56ce3b668c03014`.
- **DataOps Quality Gate (`src/data_validation.py`):** 100% Passed (7/7 Great Expectations-style invariant rules satisfied including null rate threshold, numeric physical boundaries, categorical allowable sets, exact row deduplication, and target positive invariants).

---

## 4. Feature Engineering & Strict Training-Serving Parity
Strict training-serving parity is enforced via a serialized scikit-learn `Pipeline` and shared `engineer_features()` helper:

| Feature Name | Type | Description | MLOps Rationale |
|---|---|---|---|
| `year` | Numeric | Model manufacturing year (1990–2026) | Primary chronological baseline |
| `km_driven` | Numeric | Total odometer mileage | Wear-and-tear metric |
| `vehicle_age` | Numeric | `current_year - year` | Dynamic temporal decay |
| `km_per_year` | Numeric | `km_driven / max(vehicle_age, 0.75)` | Usage intensity (heavy vs light usage) |
| `is_luxury` | Binary | 1 if OEM is luxury tier (BMW, Mercedes, Audi, etc.) | Segment-level premium indicator |
| `brand` | Categorical | Manufacturer make (Maruti, Tata, Hyundai, etc.) | One-Hot Encoded with `handle_unknown='ignore'` |
| `car_model` | Categorical | Specific model line | High-cardinality sparse encoding |
| `fuel` | Categorical | Petrol, Diesel, Electric, Hybrid, CNG, LPG | Powertrain technology indicator |
| `seller_type` | Categorical | Individual, Dealer, Trustmark Dealer | Distribution channel margin |
| `transmission`| Categorical | Manual, Automatic | Transmission mechanical premium |
| `owner` | Categorical | First Owner, Second Owner, Third Owner, etc. | Title history depreciation penalty |

---

## 5. Performance Metrics & Quality Gates (Holdout Test Set)

### Regression Performance (Selling Price in INR)
- **R² Score (Coefficient of Determination):** `0.9522` (95.2% variance explained)
- **Mean Absolute Error (MAE):** `₹ 113,704.32`
- **Median Absolute Error:** `₹ 80,188.45`
- **Root Mean Squared Error (RMSE):** `₹ 184,843.02`
- **Mean Absolute Percentage Error (MAPE):** `14.19%`

### Classification Performance (4 Market Brackets)
- **Accuracy:** `81.73%`
- **Precision (Macro):** `0.852`
- **Recall (Macro):** `0.800`
- **Macro F1-Score:** `0.8202`

---

## 6. Subgroup Slicing & Fairness Evaluation (`src/model_slicing.py`)
Model fairness evaluated across 18 subgroups to identify potential bias or performance disparity:

| Slicing Dimension | Slice Value | Sample Count | Test Share | MAE (₹) | MAPE (%) | R² Score | SLA Status |
|---|---|---|---|---|---|---|---|
| **Fuel / Powertrain** | Electric | 167 | 9.4% | ₹ 118,893 | 8.82% | 0.9589 | COMPLIANT |
| **Fuel / Powertrain** | Hybrid | 50 | 2.8% | ₹ 102,848 | 6.95% | 0.9027 | COMPLIANT |
| **Fuel / Powertrain** | Diesel | 628 | 35.3% | ₹ 126,927 | 17.42% | 0.9348 | COMPLIANT |
| **Fuel / Powertrain** | Petrol | 854 | 48.0% | ₹ 94,939 | 15.74% | 0.8851 | COMPLIANT |
| **Fuel / Powertrain** | CNG | 75 | 4.2% | ₹ 36,359 | 6.00% | 0.9086 | COMPLIANT |
| **Vehicle Age** | New (0-3 yrs) | 734 | 41.3% | ₹ 121,989 | 10.43% | 0.9584 | COMPLIANT |
| **Vehicle Age** | Mid-Age (4-7 yrs)| 401 | 22.6% | ₹ 101,084 | 12.61% | 0.9357 | COMPLIANT |
| **Vehicle Age** | Older (8+ yrs) | 643 | 36.2% | ₹ 91,123 | 21.80% | 0.8412 | COMPLIANT |
| **Market Tier** | Mass-Market | 1,720 | 96.7% | ₹ 97,942 | 14.85% | 0.9124 | COMPLIANT |
| **Market Tier** | Luxury OEM | 58 | 3.3% | ₹ 348,600 | 20.65% | 0.8810 | FLAGGED (High Variance) |

---

## 7. MLOps Lifecycle & Governance Infrastructure
- **DataOps Quality Gate (`src/data_validation.py`):** Verifies schema types, value ranges, and completeness before allowing training to proceed.
- **Model Slicing Engine (`src/model_slicing.py`):** Analyzes fairness across 18 distinct market segments.
- **Operational Telemetry & Latency Profiling (`src/telemetry.py`):** Real-time P50, P95, and P99 latency tracking with persistent audit logging in `prediction_audit.jsonl`.
- **Covariate & Prediction Drift Monitoring (`src/drift.py`):** Tracks Population Stability Index (PSI) and 2-sample Kolmogorov-Smirnov (KS) tests across both input features and output predictions.
- **Champion vs Challenger Model Registry:** Automated promotion gate comparing holdout MAE and R² against incumbent champion before serialization (`champion_metadata.json`).
- **End-to-End Continuous Training DAG (`scripts/run_pipeline.py`):** Orchestrates validation, training, slicing, drift profiling, and registry promotion with execution manifest (`pipeline_run_manifest.json`).
- **Explainable AI (XAI):** Tree-based global feature attributions serialized in `feature_importance.json`.
- **Experiment Tracking:** Full parameters, metrics, and 8 serialized governance artifacts logged to MLflow (`mlruns/`).
