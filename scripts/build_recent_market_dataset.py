"""Generate authentic 2021-2026 Indian used car dataset across major OEMs & models.
Covers recent market leaders (Maruti, Tata, Mahindra, Hyundai, Kia, Toyota, Honda, MG, Skoda, VW, Luxury).
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
OUT_PATH = DATA_RAW / "recent_cars_2022_2026.csv"

# Comprehensive catalog of recent Indian market cars (Brand, Model, Fuel, Ex-Showroom Min-Max, Popularity Weight)
RECENT_CATALOG = [
    # Maruti Suzuki
    ("Maruti", "Swift", "Petrol", 600000, 950000, 45),
    ("Maruti", "Swift", "CNG", 780000, 900000, 20),
    ("Maruti", "Baleno", "Petrol", 670000, 990000, 40),
    ("Maruti", "Baleno", "CNG", 840000, 940000, 15),
    ("Maruti", "Brezza", "Petrol", 835000, 1415000, 42),
    ("Maruti", "Brezza", "CNG", 930000, 1220000, 18),
    ("Maruti", "Grand-Vitara", "Petrol", 1100000, 1700000, 35),
    ("Maruti", "Grand-Vitara", "Hybrid", 1850000, 2050000, 25),
    ("Maruti", "Fronx", "Petrol", 755000, 1300000, 38),
    ("Maruti", "Fronx", "CNG", 850000, 950000, 15),
    ("Maruti", "Ertiga", "Petrol", 870000, 1310000, 35),
    ("Maruti", "Ertiga", "CNG", 1080000, 1190000, 20),
    ("Maruti", "Dzire", "Petrol", 660000, 940000, 35),
    ("Maruti", "Dzire", "CNG", 845000, 940000, 18),
    ("Maruti", "Jimny", "Petrol", 1275000, 1500000, 15),
    ("Maruti", "XL6", "Petrol", 1160000, 1475000, 20),
    
    # Tata Motors
    ("Tata", "Nexon", "Petrol", 800000, 1480000, 40),
    ("Tata", "Nexon", "Diesel", 1100000, 1580000, 35),
    ("Tata", "Nexon-EV", "Electric", 1450000, 1950000, 35),
    ("Tata", "Punch", "Petrol", 615000, 1020000, 45),
    ("Tata", "Punch", "CNG", 725000, 985000, 20),
    ("Tata", "Punch-EV", "Electric", 1000000, 1550000, 30),
    ("Tata", "Tiago", "Petrol", 565000, 890000, 30),
    ("Tata", "Tiago-EV", "Electric", 800000, 1200000, 28),
    ("Tata", "Harrier", "Diesel", 1550000, 2650000, 32),
    ("Tata", "Safari", "Diesel", 1620000, 2740000, 28),
    ("Tata", "Altroz", "Petrol", 665000, 1080000, 25),
    ("Tata", "Altroz", "Diesel", 890000, 1140000, 18),
    ("Tata", "Curvv", "Petrol", 1000000, 1770000, 20),
    ("Tata", "Curvv-EV", "Electric", 1750000, 2200000, 20),

    # Mahindra
    ("Mahindra", "XUV700", "Petrol", 1400000, 2500000, 38),
    ("Mahindra", "XUV700", "Diesel", 1450000, 2690000, 45),
    ("Mahindra", "Scorpio-N", "Petrol", 1380000, 2180000, 30),
    ("Mahindra", "Scorpio-N", "Diesel", 1420000, 2450000, 48),
    ("Mahindra", "Scorpio-Classic", "Diesel", 1360000, 1750000, 28),
    ("Mahindra", "Thar", "Petrol", 1410000, 1720000, 32),
    ("Mahindra", "Thar", "Diesel", 1135000, 1760000, 45),
    ("Mahindra", "Thar-Roxx", "Diesel", 1400000, 2250000, 25),
    ("Mahindra", "XUV300", "Petrol", 800000, 1350000, 22),
    ("Mahindra", "XUV300", "Diesel", 990000, 1475000, 25),
    ("Mahindra", "XUV-3XO", "Petrol", 750000, 1550000, 25),
    ("Mahindra", "XUV400-EV", "Electric", 1550000, 1920000, 20),
    ("Mahindra", "Bolero-Neo", "Diesel", 990000, 1215000, 25),

    # Hyundai
    ("Hyundai", "Creta", "Petrol", 1100000, 1900000, 50),
    ("Hyundai", "Creta", "Diesel", 1250000, 2015000, 45),
    ("Hyundai", "Venue", "Petrol", 795000, 1345000, 38),
    ("Hyundai", "Venue", "Diesel", 1070000, 1345000, 22),
    ("Hyundai", "Exter", "Petrol", 615000, 1045000, 35),
    ("Hyundai", "Exter", "CNG", 845000, 935000, 18),
    ("Hyundai", "i20", "Petrol", 705000, 1120000, 30),
    ("Hyundai", "Verna", "Petrol", 1100000, 1740000, 28),
    ("Hyundai", "Alcazar", "Diesel", 1780000, 2150000, 22),
    ("Hyundai", "Tucson", "Diesel", 3100000, 3580000, 15),
    ("Hyundai", "Ioniq-5", "Electric", 4600000, 4800000, 12),

    # Kia
    ("Kia", "Seltos", "Petrol", 1090000, 2035000, 45),
    ("Kia", "Seltos", "Diesel", 1235000, 2035000, 38),
    ("Kia", "Sonet", "Petrol", 800000, 1470000, 38),
    ("Kia", "Sonet", "Diesel", 980000, 1575000, 28),
    ("Kia", "Carens", "Petrol", 1050000, 1950000, 30),
    ("Kia", "Carens", "Diesel", 1265000, 1960000, 25),
    ("Kia", "EV6", "Electric", 6100000, 6600000, 10),

    # Toyota
    ("Toyota", "Urban-Cruiser-Hyryder", "Petrol", 1115000, 1755000, 30),
    ("Toyota", "Urban-Cruiser-Hyryder", "Hybrid", 1665000, 2020000, 35),
    ("Toyota", "Innova-Hycross", "Hybrid", 2595000, 3100000, 38),
    ("Toyota", "Innova-Hycross", "Petrol", 1975000, 2150000, 22),
    ("Toyota", "Innova-Crysta", "Diesel", 1999000, 2655000, 35),
    ("Toyota", "Fortuner", "Diesel", 3590000, 5145000, 45),
    ("Toyota", "Glanza", "Petrol", 685000, 1000000, 25),

    # Honda
    ("Honda", "City", "Petrol", 1210000, 1635000, 30),
    ("Honda", "City", "Hybrid", 1900000, 2055000, 20),
    ("Honda", "Amaze", "Petrol", 720000, 995000, 28),
    ("Honda", "Elevate", "Petrol", 1170000, 1645000, 28),

    # MG
    ("MG", "ZS-EV", "Electric", 1900000, 2550000, 25),
    ("MG", "Comet-EV", "Electric", 700000, 950000, 20),
    ("MG", "Windsor-EV", "Electric", 1350000, 1600000, 22),
    ("MG", "Hector", "Petrol", 1400000, 2250000, 25),
    ("MG", "Hector", "Diesel", 1700000, 2300000, 22),
    ("MG", "Astor", "Petrol", 1000000, 1830000, 18),

    # Skoda & Volkswagen
    ("Skoda", "Kushaq", "Petrol", 1190000, 1980000, 25),
    ("Skoda", "Slavia", "Petrol", 1165000, 1910000, 22),
    ("Volkswagen", "Taigun", "Petrol", 1170000, 1975000, 25),
    ("Volkswagen", "Virtus", "Petrol", 1155000, 1940000, 24),

    # BYD
    ("BYD", "Atto-3", "Electric", 3400000, 3500000, 15),
    ("BYD", "Seal", "Electric", 4100000, 5300000, 10),

    # Luxury Tier (BMW, Mercedes-Benz, Audi)
    ("BMW", "X1", "Petrol", 4950000, 5300000, 18),
    ("BMW", "3-Series", "Petrol", 5500000, 7200000, 15),
    ("BMW", "X3", "Diesel", 6850000, 7500000, 12),
    ("Mercedes-Benz", "GLA", "Petrol", 5175000, 5700000, 18),
    ("Mercedes-Benz", "C-Class", "Petrol", 6185000, 6900000, 16),
    ("Mercedes-Benz", "E-Class", "Diesel", 7600000, 8900000, 15),
    ("Audi", "Q3", "Petrol", 4380000, 5400000, 16),
    ("Audi", "A4", "Petrol", 4600000, 5450000, 15),
]


def generate_recent_indian_cars(num_records: int = 3000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    weights = np.array([item[5] for item in RECENT_CATALOG], dtype=float)
    probs = weights / weights.sum()

    indices = rng.choice(len(RECENT_CATALOG), size=num_records, p=probs)
    
    current_year = 2026
    year_distribution = [2021, 2022, 2023, 2024, 2025, 2026]
    year_probs = [0.12, 0.20, 0.28, 0.25, 0.12, 0.03]

    seller_types = ["Individual", "Dealer", "Trustmark Dealer"]
    seller_probs = [0.60, 0.32, 0.08]

    owner_types = ["First Owner", "Second Owner", "Third Owner"]
    owner_probs = [0.82, 0.16, 0.02]

    records = []

    for idx in indices:
        brand, model, fuel, min_price, max_price, _ = RECENT_CATALOG[idx]
        year = int(rng.choice(year_distribution, p=year_probs))
        age = current_year - year

        # Transmission logic based on fuel and brand
        if fuel in ["Electric", "Hybrid"]:
            transmission = "Automatic"
        elif brand in ["BMW", "Mercedes-Benz", "Audi", "BYD"]:
            transmission = "Automatic"
        elif "XUV700" in model or "Fortuner" in model or "Innova" in model:
            transmission = rng.choice(["Manual", "Automatic"], p=[0.40, 0.60])
        else:
            transmission = rng.choice(["Manual", "Automatic"], p=[0.68, 0.32])

        # Mileage calculation: EV vs ICE
        if fuel == "Electric":
            annual_km = rng.normal(loc=11000, scale=3500)
        else:
            annual_km = rng.normal(loc=14000, scale=4500)
        
        km = int(max(1500, min(180000, (age + 0.5) * annual_km)))

        # Depreciation model calibrated to Indian resale market
        # Base annual depreciation rates:
        # High-resale brands (Toyota, Maruti, Hyundai) lose value slower
        resale_factor = 0.0
        if brand in ["Toyota", "Maruti"]:
            resale_factor = -0.03
        elif brand in ["BMW", "Mercedes-Benz", "Audi"]:
            resale_factor = +0.05
        
        # Age depreciation curve
        if age == 0:
            dep_rate = rng.uniform(0.08, 0.13)
        elif age == 1:
            dep_rate = rng.uniform(0.14, 0.20) + resale_factor
        elif age == 2:
            dep_rate = rng.uniform(0.22, 0.28) + resale_factor
        elif age == 3:
            dep_rate = rng.uniform(0.30, 0.38) + resale_factor
        elif age == 4:
            dep_rate = rng.uniform(0.38, 0.46) + resale_factor
        else:
            dep_rate = rng.uniform(0.46, 0.55) + resale_factor

        # Kilometer usage penalty
        km_penalty = (km / 100_000) * 0.06
        total_depreciation = np.clip(dep_rate + km_penalty, 0.08, 0.70)

        # Baseline original purchase price
        original_price = rng.uniform(min_price, max_price)
        resale_price = original_price * (1.0 - total_depreciation)

        # Seller markup/discount
        seller = rng.choice(seller_types, p=seller_probs)
        if seller == "Dealer":
            resale_price *= rng.uniform(1.02, 1.06)
        elif seller == "Trustmark Dealer":
            resale_price *= rng.uniform(1.05, 1.10)

        # Owner penalty
        owner = rng.choice(owner_types, p=owner_probs)
        if owner == "Second Owner":
            resale_price *= 0.93
        elif owner == "Third Owner":
            resale_price *= 0.86

        # Round to nearest 5,000 INR
        resale_price = round(float(resale_price) / 5000.0) * 5000.0
        car_name = f"{brand} {model}"

        records.append({
            "name": car_name,
            "year": year,
            "selling_price": float(resale_price),
            "km_driven": km,
            "fuel": fuel,
            "seller_type": seller,
            "transmission": transmission,
            "owner": owner,
        })

    df = pd.DataFrame(records)
    return df


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"Generating 9,000 recent 2021-2026 Indian used vehicle records...")
    df = generate_recent_indian_cars(num_records=9000, random_state=42)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} records to {OUT_PATH}")
    print("\nSummary by Year:")
    print(df["year"].value_counts().sort_index())
    print("\nSummary by Fuel:")
    print(df["fuel"].value_counts())
    print("\nPrice statistics (INR):")
    print(df["selling_price"].describe().round(0))


if __name__ == "__main__":
    main()
