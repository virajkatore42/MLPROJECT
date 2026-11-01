# Model Card: India Used Car & EV Resale Price Prediction Suite (SDG-9)

## 1. Model Details
- **Model Name:** AutoValuate AI Resale Price Regressor & Bracket Classifier
- **Model Version:** `2.1.0-ct-ensemble`
- **Model Type:** Dual-Architecture System:
  1. **Continuous Regressor:** `TransformedTargetRegressor(HistGradientBoostingRegressor)` with dynamic logarithmic target scaling (`func=np.log1p, inverse_func=np.expm1`).
  2. **Tier Classifier:** `RandomForestClassifier` with balanced class weights predicting 4 price brackets (`Budget`, `Mid-Range`, `Premium`, `Luxury`).
- **Release Date:** November 2026
- **License:** MIT / Academic Production

---

## 2. Intended Use & Domain
- **Primary Use Case:** Transparent, data-driven price estimation for pre-owned passenger vehicles and electric vehicles (EVs) in the Indian automotive market.
- **Target Audience:** Individual buyers/sellers, pre-owned car dealerships, certified refurbishment centers, auto financing lenders.
- **Out-of-Scope Uses:** Commercial fleet heavy-duty vehicles (buses/trucks), salvaged/total-loss insurance write-offs, exotic collector supercars (> ₹1 Crore).

---

## 3. Training Data & Multi-Source Lineage (DataOps)
The dataset represents a multi-source, continuous ingestion pipeline spanning **1992 through 2026**:
1. **Historical CarDekho Dataset:** 4,340 historical listings (1992–2020).
2. **Indian Electric Vehicle (EV) Dataset:** Authentic & augmented EV listings (Tata Nexon-EV, Punch-EV, Tiago-EV, Curvv-EV, MG ZS-EV, Comet-EV, Windsor-EV, Mahindra XUV400, BYD Atto-3/Seal, Hyundai Ioniq-5, Kia EV6).
3. **Modern 2021–2026 Indian Market Dataset:** 5,000 recent listings covering high-volume market leaders (Maruti Swift, Brezza, Grand Vitara, Fronx; Hyundai Creta, Venue, Exter; Mahindra XUV700, Thar, Scorpio-N; Toyota Hyryder, Innova Hycross; Kia Seltos, Sonet).
- **Total Ingested Listings:** 8,888 clean automotive records.
- **Lineage Fingerprint:** SHA-256 hash `8940998832d7eff9`.

---

## 4. Feature Engineering & Training-Serving Parity
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
- **R² Score (Coefficient of Determination):** `0.9188` (91.9% variance explained)
- **Mean Absolute Error (MAE):** `₹ 106,541.43`
- **Median Absolute Error:** `₹ 69,468.35`
- **Root Mean Squared Error (RMSE):** `₹ 220,483.91`
- **Mean Absolute Percentage Error (MAPE):** `15.02%`

### Classification Performance (4 Market Brackets)
- **Accuracy:** `80.76%`
- **Precision (Macro):** `0.803`
- **Recall (Macro):** `0.812`
- **Macro F1-Score:** `0.8071`

---

## 6. MLOps Lifecycle & Governance Infrastructure
- **Data Drift Detection (`src/drift.py`):** Calculates Population Stability Index (PSI) and 2-sample Kolmogorov-Smirnov (KS) tests between live inference probes and serialized training baselines (`drift_baseline.json`).
- **Explainable AI (`src/explainability.py`):** Global feature attributions extracted and tracked in `feature_importance.json`.
- **Champion vs Challenger Framework:** Automated comparison in `src/train.py` promoting challenger models only when holdout metrics meet SLA performance gates (`champion_metadata.json`).
- **Continuous Training (CT):** Executable trigger `scripts/retrain.py` for automated scheduled or drift-triggered model updates.
- **Experiment Tracking:** Full run metrics, parameters, tags, and serialized artifacts logged to MLflow (`mlruns/`).
- **Training-Serving Parity:** Identical preprocessing graph serialized into `price_model.joblib` with zero data leakage.

---

## 7. SDG-9 Alignment & Ethical Considerations
- **SDG-9 Goal:** *Industry, Innovation, and Infrastructure*.
- **Market Transparency:** Removes informational asymmetry in India's pre-owned automotive sector, preventing predatory pricing by intermediaries.
- **Green Transition:** Quantifies Total Cost of Ownership (TCO) savings and lifecycle carbon reduction for Electric Vehicles (EVs) and Hybrids, accelerating zero-emission vehicle adoption.
