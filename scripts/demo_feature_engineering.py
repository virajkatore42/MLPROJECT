"""Demo script: Automating Feature Engineering in MLOps.
Runs live in terminal to demonstrate:
1. Automated domain extraction & string parsing from raw listings.
2. Dynamic temporal feature derivation (vehicle_age, km_per_year, is_luxury).
3. Automated ColumnTransformer pipeline (scaling + one-hot encoding).
4. Edge-case handling (unseen categories without crash).
5. Training-Serving Parity using the serialized pipeline artifact.
"""

import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_PATH
from src.data_prep import (
    assign_price_bracket,
    engineer_features,
    extract_brand,
    extract_car_model,
)


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def main():
    print_banner("MLOps Automated Feature Engineering Demo")
    print("Project: India Used Car & EV Valuation Suite (SDG-9)")
    print(f"Model Artifact: {MODEL_PATH.name}")

    # STAGE 1: Automated Raw Text Parsing & Domain Extraction
    print_banner("Stage 1: Automated Text Parsing & Domain Extraction")
    raw_car_names = [
        "Maruti Swift Dzire VDI BSIV",
        "Tata Nexon EV XZ Plus Lux",
        "Hyundai Grand i10 Magna",
        "MG ZS EV Excite 2022",
    ]
    print(f"{'Raw Input Listing String':<35} | {'Extracted Brand':<15} | {'Extracted Model':<15}")
    print("-" * 70)
    for name in raw_car_names:
        brand = extract_brand(name)
        model = extract_car_model(name)
        print(f"{name:<35} | {brand:<15} | {model:<15}")

    # STAGE 2: Automated Dynamic Feature Derivation
    print_banner("Stage 2: Dynamic Temporal & Discretization Derivations")
    sample_year = 2022
    current_year = datetime.now().year
    derived_age = current_year - sample_year
    sample_price = 1150000.0
    derived_bracket = assign_price_bracket(sample_price)

    print(f"-> Raw Input Year: {sample_year}")
    print(f"-> Dynamic System Clock Year: {current_year}")
    print(f"-> Automatically Engineered Feature [vehicle_age]: {derived_age} years")
    print(f"-> Continuous Target: Rs. {sample_price:,.0f}")
    print(f"-> Discretized Target [price_bracket]: '{derived_bracket}' (Auto-binned for classification)")

    # STAGE 3: Pipeline Vectorization (ColumnTransformer)
    print_banner("Stage 3: Automated Pipeline Vectorization & Scaling")
    if not MODEL_PATH.exists():
        print(f"Error: {MODEL_PATH} not found. Run python -m src.train first.")
        return

    pipeline = joblib.load(MODEL_PATH)
    preprocessor = pipeline.named_steps["prep"]

    sample_raw = pd.DataFrame([{
        "year": 2022,
        "km_driven": 25000,
        "brand": "Tata",
        "car_model": "Nexon-EV",
        "fuel": "Electric",
        "seller_type": "Individual",
        "transmission": "Automatic",
        "owner": "First Owner",
    }])

    # Apply automated feature engineering
    sample_engineered = engineer_features(sample_raw, current_year=current_year)

    print("Engineered Input Vector:")
    for col, val in sample_engineered.iloc[0].items():
        print(f"  - {col:<15}: {val}")

    transformed_vector = preprocessor.transform(sample_engineered)
    print(f"\n[Automated Transformation Result]")
    print(f"-> Output Feature Vector Dimension: {transformed_vector.shape[1]} features")
    print("-> Automated Operations Applied in One Step:")
    print("  1. StandardScaler -> mean-centered and unit-variance scaled numeric features")
    print("  2. OneHotEncoder  -> mapped high-cardinality brand, model, fuel, owner to sparse binary space")
    print(f"-> Transformed Features Preview: {transformed_vector[0][:8].round(3).tolist()}...")

    # STAGE 4: Production Edge Case: Handling Unseen Categories
    print_banner("Stage 4: Automated Edge-Case Handling (Zero Downtime)")
    unseen_raw = pd.DataFrame([{
        "year": 2024,
        "km_driven": 8000,
        "brand": "Tesla",
        "car_model": "Cybertruck",
        "fuel": "Electric",
        "seller_type": "Individual",
        "transmission": "Automatic",
        "owner": "First Owner",
    }])
    unseen_engineered = engineer_features(unseen_raw, current_year=current_year)

    print("Testing with brand='Tesla', car_model='Cybertruck' (NEVER seen in training data)...")
    try:
        unseen_vector = preprocessor.transform(unseen_engineered)
        unseen_pred = pipeline.predict(unseen_engineered)[0]
        print("SUCCESS: Pipeline auto-handled unseen category via handle_unknown='ignore'.")
        print(f"-> Output Dimension: {unseen_vector.shape[1]} features (zero-filled unknown columns)")
        print(f"-> Model Prediction without crash: Rs. {unseen_pred:,.2f}")
    except Exception as e:
        print(f"FAILED with error: {e}")

    # STAGE 5: Training-Serving Parity
    print_banner("Stage 5: Training-Serving Parity in Production")
    print("Summary of MLOps Benefits:")
    print("  1. ZERO Data Leakage: Scaler was fitted strictly on X_train.")
    print("  2. ZERO Code Duplication: FastAPI & Streamlit call engineer_features() + pipeline.predict().")
    print("  3. 100% Deterministic: Preprocessing graph is serialized inside price_model.joblib.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
