"""Statistical valuation intervals and residual depreciation projections for pre-owned vehicles.
"""


def calculate_negotiation_tiers(predicted_price: float) -> dict:
    """Calculates empirical statistical pricing bandwidths for market transactions."""
    liquidation = round(predicted_price * 0.90, -2)       # Quick liquidation boundary (-10%)
    fair_market = round(predicted_price, -2)               # Central point prediction
    premium_listing = round(predicted_price * 1.08, -2)   # Certified/verified vehicle boundary (+8%)

    return {
        "instant_cash_price": float(liquidation),
        "fair_market_price": float(fair_market),
        "patient_seller_price": float(premium_listing),
        "dealer_margin_gap": float(fair_market - liquidation),
    }


def forecast_depreciation(current_price: float, vehicle_age: int, fuel: str = "Petrol") -> list[dict]:
    """Generates 12-month future resale depreciation projection based on age curve."""
    # Annual depreciation rate factor based on age
    if vehicle_age <= 3:
        annual_deprec = 0.12
    elif vehicle_age <= 6:
        annual_deprec = 0.10
    else:
        annual_deprec = 0.08

    projection = []
    milestones = [
        ("Current", 0, 0.0),
        ("In 3 Months", 3, annual_deprec * (3 / 12)),
        ("In 6 Months", 6, annual_deprec * (6 / 12)),
        ("In 9 Months", 9, annual_deprec * (9 / 12)),
        ("In 12 Months", 12, annual_deprec),
    ]

    for label, months, deprec_pct in milestones:
        projected_val = round(current_price * (1.0 - deprec_pct), -2)
        projection.append({
            "period": label,
            "months_ahead": months,
            "projected_price_inr": float(projected_val),
            "projected_price_lakh": round(projected_val / 100000, 2),
            "value_retention_pct": round((1.0 - deprec_pct) * 100, 1),
        })

    return projection
