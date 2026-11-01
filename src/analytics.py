"""Analytical engine for 3-Tier Negotiation, 12-Month Depreciation Forecast,
EV Running Cost/TCO comparison, and Repair ROI simulation.
"""

def calculate_negotiation_tiers(predicted_price: float) -> dict:
    """Calculates 3-tier negotiation bandwidths for realistic buyer/seller bargaining."""
    instant_cash = round(predicted_price * 0.90, -2)       # Instant 24h dealer liquidation (-10%)
    fair_market = round(predicted_price, -2)               # Fair peer-to-peer market value
    patient_seller = round(predicted_price * 1.08, -2)     # Patient seller with complete service history (+8%)
    
    return {
        "instant_cash_price": float(instant_cash),
        "fair_market_price": float(fair_market),
        "patient_seller_price": float(patient_seller),
        "dealer_margin_gap": float(fair_market - instant_cash),
    }


def forecast_depreciation(current_price: float, vehicle_age: int, fuel: str) -> list[dict]:
    """Generates a 12-month future resale depreciation projection."""
    # Annual depreciation rate factor based on age and fuel type
    if fuel.lower() == "electric":
        annual_deprec = 0.12 if vehicle_age <= 3 else 0.14
    elif fuel.lower() == "diesel":
        annual_deprec = 0.11 if vehicle_age <= 4 else 0.15
    else:  # Petrol / CNG
        annual_deprec = 0.09 if vehicle_age <= 4 else 0.12

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


def calculate_tco_comparison(annual_km: int = 12000, years: int = 3) -> dict:
    """Compares 3-Year running cost of Electric vs Petrol vs Diesel in India."""
    total_km = annual_km * years

    # Electricity cost: ₹7.5/unit, ~8km per unit = ~₹0.95 to ₹1.25/km
    ev_fuel_cost = total_km * 1.20
    ev_service_cost = years * 6000  # Lower mechanical service (no engine oil, filters)
    ev_total = ev_fuel_cost + ev_service_cost

    # Petrol cost: ₹102/litre, ~14kmpl = ~₹7.30/km
    petrol_fuel_cost = total_km * 7.30
    petrol_service_cost = years * 11000
    petrol_total = petrol_fuel_cost + petrol_service_cost

    # Diesel cost: ₹90/litre, ~17kmpl = ~₹5.30/km
    diesel_fuel_cost = total_km * 5.30
    diesel_service_cost = years * 14000
    diesel_total = diesel_fuel_cost + diesel_service_cost

    savings_vs_petrol = petrol_total - ev_total
    savings_vs_diesel = diesel_total - ev_total

    return {
        "evaluation_period_years": years,
        "total_km_evaluated": total_km,
        "electric_running_total": float(round(ev_total)),
        "petrol_running_total": float(round(petrol_total)),
        "diesel_running_total": float(round(diesel_total)),
        "ev_savings_vs_petrol": float(round(savings_vs_petrol)),
        "ev_savings_vs_diesel": float(round(savings_vs_diesel)),
    }


def simulate_repair_roi(base_price: float, new_tires: bool = False, paint_touchup: bool = False, full_detailing: bool = False) -> dict:
    """Calculates the net ROI on pre-sale vehicle touch-ups."""
    repairs = []
    total_cost = 0
    total_boost = 0

    if new_tires:
        cost, boost = 14000, 24000
        total_cost += cost
        total_boost += boost
        repairs.append({"item": "New Set of Tires", "cost": cost, "estimated_value_boost": boost, "net_gain": boost - cost})

    if paint_touchup:
        cost, boost = 6000, 11000
        total_cost += cost
        total_boost += boost
        repairs.append({"item": "Minor Dent & Bumper Paint Touch-Up", "cost": cost, "estimated_value_boost": boost, "net_gain": boost - cost})

    if full_detailing:
        cost, boost = 3500, 7500
        total_cost += cost
        total_boost += boost
        repairs.append({"item": "Professional Deep Interior Detailing", "cost": cost, "estimated_value_boost": boost, "net_gain": boost - cost})

    optimized_price = base_price + total_boost
    net_profit = total_boost - total_cost

    return {
        "original_price": float(base_price),
        "total_repair_investment": float(total_cost),
        "estimated_value_boost": float(total_boost),
        "optimized_price": float(optimized_price),
        "net_profit": float(net_profit),
        "details": repairs,
    }
