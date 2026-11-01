from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "cardekho.csv"
DATA_V4 = PROJECT_ROOT / "data" / "raw" / "car_details_v4.csv"
DATA_EV = PROJECT_ROOT / "data" / "raw" / "ev_cars.csv"
DATA_RECENT = PROJECT_ROOT / "data" / "raw" / "recent_cars_2022_2026.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "price_model.joblib"
CLASSIFIER_MODEL_PATH = MODEL_DIR / "classifier_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
DRIFT_BASELINE_PATH = MODEL_DIR / "drift_baseline.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"
CHAMPION_METADATA_PATH = MODEL_DIR / "champion_metadata.json"
MODEL_CARD_PATH = PROJECT_ROOT / "MODEL_CARD.md"

RANDOM_STATE = 42
CURRENT_YEAR = 2026
TARGET = "selling_price"
CLASSIFICATION_TARGET = "price_bracket"
BRACKET_LABELS = ["Budget", "Mid-Range", "Premium", "Luxury"]

LUXURY_BRANDS = {
    "BMW",
    "Mercedes-Benz",
    "Audi",
    "Jaguar",
    "Land Rover",
    "Volvo",
    "Porsche",
    "Lexus",
    "Maserati",
    "Ferrari",
    "Rolls-Royce",
    "Bentley",
    "Mini",
    "MINI",
}

NUMERIC_FEATURES = [
    "year",
    "km_driven",
    "vehicle_age",
    "km_per_year",
    "is_luxury",
]

CATEGORICAL_FEATURES = [
    "brand",
    "car_model",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]
