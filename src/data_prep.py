from pathlib import Path
import numpy as np
import pandas as pd

from src.config import CURRENT_YEAR, LUXURY_BRANDS


def extract_brand(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return "Unknown"
    brand = name.strip().split()[0]
    if brand.upper() == "MARUTI" or "MARUTI" in brand.upper():
        return "Maruti"
    if brand.upper() == "MINI":
        return "Mini"
    return brand


def extract_car_model(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return "Other"
    parts = name.strip().split()
    if len(parts) > 2 and parts[2].upper() == "EV":
        return f"{parts[1]}-EV"
    elif len(parts) > 1:
        clean = parts[1].replace("/", "-")
        return clean
    return "Other"


def assign_price_bracket(price: float) -> str:
    if price < 300_000:
        return "Budget"
    elif price < 800_000:
        return "Mid-Range"
    elif price < 1_500_000:
        return "Premium"
    else:
        return "Luxury"


def get_brand_model_mapping(df: pd.DataFrame) -> dict[str, list[str]]:
    mapping = {}
    grouped = df.groupby("brand")["car_model"].unique()
    for brand, models in grouped.items():
        sorted_models = sorted([m for m in models if m and m != "Other"])
        if not sorted_models:
            sorted_models = ["Other"]
        mapping[str(brand)] = sorted_models
    return mapping


def engineer_features(df: pd.DataFrame, current_year: int = CURRENT_YEAR) -> pd.DataFrame:
    """Automate feature derivations to ensure strict Training-Serving parity."""
    res = df.copy()

    # Dynamic temporal derivation
    res["year"] = pd.to_numeric(res["year"], errors="coerce").fillna(current_year).astype(int)
    res["km_driven"] = pd.to_numeric(res["km_driven"], errors="coerce").fillna(45000.0).astype(float)
    
    res["vehicle_age"] = (current_year - res["year"]).clip(lower=0).astype(int)
    res["km_per_year"] = (res["km_driven"] / np.maximum(res["vehicle_age"], 0.75)).round(1)

    # Market tier indicator
    if "brand" in res.columns:
        res["brand"] = res["brand"].astype(str).str.strip().map(lambda b: "Maruti" if "MARUTI" in b.upper() else b)
        res["is_luxury"] = res["brand"].isin(LUXURY_BRANDS).astype(int)
    else:
        res["is_luxury"] = 0

    # Clean categorical text
    for col in ["fuel", "seller_type", "transmission", "owner", "brand", "car_model"]:
        if col in res.columns:
            res[col] = res[col].astype(str).str.strip()

    return res


def load_and_clean(path: str | Path | None = None, include_v4: bool = False) -> pd.DataFrame:
    """Multi-source ingestion pipeline merging historical CarDekho, modern EVs, and 2021-2026 market streams."""
    if path is None:
        from src.config import DATA_RAW
        primary_path = Path(DATA_RAW)
    else:
        primary_path = Path(path)

    data_dir = primary_path.parent
    dfs = []

    # 1. Primary CarDekho historical listings (1992-2020)
    if primary_path.exists():
        try:
            df_primary = pd.read_csv(primary_path)
            dfs.append(df_primary)
        except Exception:
            pass

    # 2. Modern authentic & augmented Indian EV dataset (2020-2026)
    ev_path = data_dir / "ev_cars.csv"
    if ev_path.exists() and ev_path.resolve() != primary_path.resolve():
        try:
            df_ev = pd.read_csv(ev_path)
            dfs.append(df_ev)
        except Exception:
            pass

    # 3. Modern recent 2021-2026 Indian market listings
    recent_path = data_dir / "recent_cars_2022_2026.csv"
    if recent_path.exists() and recent_path.resolve() != primary_path.resolve():
        try:
            df_recent = pd.read_csv(recent_path)
            dfs.append(df_recent)
        except Exception:
            pass

    # 4. Optional: Consumer V4 listings under 1.8M INR
    if include_v4:
        v4_path = data_dir / "car_details_v4.csv"
        if v4_path.exists() and v4_path.resolve() != primary_path.resolve():
            try:
                df_v4 = pd.read_csv(v4_path)
                v4_mapped = pd.DataFrame()
                v4_mapped["name"] = df_v4["Make"].astype(str).str.strip() + " " + df_v4["Model"].astype(str).str.strip()
                v4_mapped["year"] = df_v4["Year"]
                v4_mapped["selling_price"] = df_v4["Price"]
                v4_mapped["km_driven"] = df_v4["Kilometer"]
                v4_mapped["fuel"] = df_v4["Fuel Type"].replace({
                    "CNG + CNG": "CNG",
                    "Petrol + CNG": "CNG",
                    "Petrol + LPG": "LPG",
                })
                v4_mapped["seller_type"] = df_v4["Seller Type"].replace({
                    "Corporate": "Dealer",
                    "Commercial Registration": "Dealer",
                })
                v4_mapped["transmission"] = df_v4["Transmission"]
                v4_mapped["owner"] = df_v4["Owner"].replace({
                    "First": "First Owner",
                    "Second": "Second Owner",
                    "Third": "Third Owner",
                    "Fourth": "Fourth & Above Owner",
                    "4 or More": "Fourth & Above Owner",
                    "UnRegistered Car": "Test Drive Car",
                })
                v4_mapped = v4_mapped[v4_mapped["selling_price"] <= 1_800_000]
                dfs.append(v4_mapped)
            except Exception:
                pass

    if not dfs:
        raise FileNotFoundError(f"No valid dataset could be loaded from {data_dir}")

    # Standardize and concatenate
    combined = pd.concat(dfs, ignore_index=True)
    combined.columns = [c.strip().lower().replace(" ", "_") for c in combined.columns]

    rename = {
        "name": "car_name",
        "selling_price": "selling_price",
        "km_driven": "km_driven",
    }
    combined = combined.rename(columns={k: v for k, v in rename.items() if k in combined.columns})

    # Data hygiene & boundary validation (DataOps Invariants)
    combined = combined.dropna(subset=["selling_price", "year", "km_driven"])
    combined["year"] = pd.to_numeric(combined["year"], errors="coerce")
    combined["km_driven"] = pd.to_numeric(combined["km_driven"], errors="coerce")
    combined["selling_price"] = pd.to_numeric(combined["selling_price"], errors="coerce")
    combined = combined.dropna(subset=["selling_price", "year", "km_driven"])

    # Filtering realistic automotive bounds
    combined = combined[(combined["year"] >= 1990) & (combined["year"] <= CURRENT_YEAR)]
    combined = combined[(combined["selling_price"] >= 20_000) & (combined["selling_price"] <= 12_000_000)]
    combined = combined[(combined["km_driven"] >= 0) & (combined["km_driven"] <= 400_000)]

    # Parse brand and model
    if "car_name" in combined.columns:
        combined["brand"] = combined["car_name"].map(extract_brand)
        combined["car_model"] = combined["car_name"].map(extract_car_model)
    else:
        combined["brand"] = "Unknown"
        combined["car_model"] = "Other"

    # Harmonize brand names and owner categories
    combined["brand"] = combined["brand"].replace({
        "Maruti Suzuki": "Maruti",
        "MINI": "Mini",
    })
    owner_harmonize = {
        "First": "First Owner",
        "Second": "Second Owner",
        "Third": "Third Owner",
        "Fourth": "Fourth & Above Owner",
        "4 or More": "Fourth & Above Owner",
        "UnRegistered Car": "Test Drive Car",
    }
    if "owner" in combined.columns:
        combined["owner"] = combined["owner"].replace(owner_harmonize)

    # Discretized price bracket target
    combined["price_bracket"] = combined["selling_price"].map(assign_price_bracket)

    # Feature engineering
    clean_df = engineer_features(combined, current_year=CURRENT_YEAR)

    # Deduplicate exact duplicate rows across all features and target
    features_and_target = [
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
    existing_cols = [c for c in features_and_target if c in clean_df.columns]
    clean_df = clean_df.drop_duplicates(subset=existing_cols).reset_index(drop=True)

    return clean_df
