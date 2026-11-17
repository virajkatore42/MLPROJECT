"""Statistical valuation intervals, residual depreciation projections, value enhancement ROI,
and market timing intelligence for pre-owned vehicles.
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


def calculate_value_enhancements(
    base_price: float,
    fuel: str = "Petrol",
    km_driven: int = 30000,
    vehicle_age: int = 3,
) -> list[dict]:
    """Calculates high-ROI modifications and detailing prep actions that improve resale value.

    Returns itemized catalog with cost, value boost, net profit, and ROI multiplier.
    """
    enhancements = []
    is_ev_or_hybrid = fuel.strip().lower() in ["electric", "hybrid"]

    # 1. OEM Service Logbook Stamp
    service_boost = min(45000.0, max(20000.0, round(base_price * 0.035, -2)))
    enhancements.append({
        "id": "oem_service_record",
        "title": "📋 Certified OEM Service Logbook & Stamp",
        "description": "Authorized dealer service records proving scheduled maintenance and authentic odometer mileage.",
        "cost_inr": 1500.0,
        "boost_inr": float(service_boost),
        "net_roi_inr": float(service_boost - 1500.0),
        "roi_multiplier": round(service_boost / 1500.0, 1),
        "category": "Documentation & Trust",
    })

    # 2. Stage-2 Ceramic Detailing & Paint Correction
    ceramic_boost = min(35000.0, max(18000.0, round(base_price * 0.026, -2)))
    enhancements.append({
        "id": "ceramic_coating",
        "title": "💎 Stage-2 Paint Correction & Ceramic Detailing",
        "description": "Removes clear-coat swirl marks, oxidization, and restores mirror showroom gloss for inspection.",
        "cost_inr": 7500.0,
        "boost_inr": float(ceramic_boost),
        "net_roi_inr": float(ceramic_boost - 7500.0),
        "roi_multiplier": round(ceramic_boost / 7500.0, 1),
        "category": "Exterior Curb Appeal",
    })

    # 3. Paintless Dent Removal & Bumper Scuff Touchup
    pdr_boost = min(25000.0, max(12000.0, round(base_price * 0.018, -2)))
    enhancements.append({
        "id": "dent_scratch_repair",
        "title": "🔨 Paintless Dent Removal (PDR) & Scuff Touchup",
        "description": "Eliminates visible door dings and bumper abrasions used by buyers as aggressive bargaining chips.",
        "cost_inr": 4500.0,
        "boost_inr": float(pdr_boost),
        "net_roi_inr": float(pdr_boost - 4500.0),
        "roi_multiplier": round(pdr_boost / 4500.0, 1),
        "category": "Exterior Condition",
    })

    # 4. Interior Deep Steam Clean & Leather Conditioning
    interior_boost = min(20000.0, max(10000.0, round(base_price * 0.015, -2)))
    enhancements.append({
        "id": "interior_steam_detailing",
        "title": "🧽 Interior Deep Steam Sanitation & Conditioning",
        "description": "Deodorizes AC vents, extracts fabric stains, and details dash/leather to factory fresh feel.",
        "cost_inr": 3200.0,
        "boost_inr": float(interior_boost),
        "net_roi_inr": float(interior_boost - 3200.0),
        "roi_multiplier": round(interior_boost / 3200.0, 1),
        "category": "Cabin & Interior",
    })

    # 5. Battery State-of-Health (EV/Hybrid) vs Fresh 12V Battery (ICE)
    if is_ev_or_hybrid:
        battery_boost = min(60000.0, max(30000.0, round(base_price * 0.042, -2)))
        enhancements.append({
            "id": "battery_certification",
            "title": "⚡ High-Voltage Battery Health (SoH) Diagnostic Certificate",
            "description": "Official EV diagnostic report confirming battery health > 88%, eliminating buyer replacement anxiety.",
            "cost_inr": 3500.0,
            "boost_inr": float(battery_boost),
            "net_roi_inr": float(battery_boost - 3500.0),
            "roi_multiplier": round(battery_boost / 3500.0, 1),
            "category": "Powertrain Certification",
        })
    else:
        battery_boost = min(18000.0, max(9000.0, round(base_price * 0.014, -2)))
        enhancements.append({
            "id": "battery_12v",
            "title": "🔋 Fresh High-Crank 12V Battery with Warranty Card",
            "description": "New warranty-backed battery providing instant single-turn crank during buyer test drives.",
            "cost_inr": 4200.0,
            "boost_inr": float(battery_boost),
            "net_roi_inr": float(battery_boost - 4200.0),
            "roi_multiplier": round(battery_boost / 4200.0, 1),
            "category": "Powertrain & Electrical",
        })

    # 6. Active Insurance & Transfer Readiness
    ins_boost = min(22000.0, max(12000.0, round(base_price * 0.018, -2)))
    enhancements.append({
        "id": "insurance_transfer",
        "title": "🛡️ Active Comprehensive Zero-Depreciation Insurance Ready",
        "description": "Transferable policy with verified NCB (No Claim Bonus) ensuring instant buyer road legality.",
        "cost_inr": 2200.0,
        "boost_inr": float(ins_boost),
        "net_roi_inr": float(ins_boost - 2200.0),
        "roi_multiplier": round(ins_boost / 2200.0, 1),
        "category": "Documentation & Trust",
    })

    return enhancements


def simulate_selling_timing(
    base_price: float,
    vehicle_age: int,
    km_driven: int,
    fuel: str = "Petrol",
    months_to_hold: int = 3,
    season_type: str = "normal",
    km_per_month: int = 1000,
) -> dict:
    """Simulates market value if selling later, incorporating seasonal demand and mileage cliffs.

    Parameters
    ----------
    base_price : Current fair market valuation in INR.
    vehicle_age : Age of vehicle in years.
    km_driven : Current odometer reading.
    fuel : Fuel type ('Petrol', 'Diesel', 'Electric', 'Hybrid', 'CNG').
    months_to_hold : Number of months to wait before selling (0 to 12).
    season_type : 'normal', 'festive' (Diwali/Navratri surge), or 'year_end' (Dec clearance discount).
    km_per_month : Estimated distance driven per month while waiting.
    """
    if months_to_hold <= 0:
        return {
            "months_to_hold": 0,
            "projected_price_inr": float(base_price),
            "projected_price_lakh": round(base_price / 100000, 2),
            "delta_inr": 0.0,
            "delta_lakh": 0.0,
            "pct_change": 0.0,
            "recommendation": "IMMEDIATE SALE",
            "verdict_title": "Sell Today at Peak Value",
            "rationale": "Immediate liquidation secures current certified market price without exposure to depreciation or mileage wear.",
        }

    # 1. Base Age Depreciation
    if vehicle_age <= 3:
        annual_deprec = 0.12
    elif vehicle_age <= 6:
        annual_deprec = 0.10
    else:
        annual_deprec = 0.08
    monthly_deprec = (annual_deprec / 12.0) * months_to_hold

    # 2. Mileage & Psychological Cliff Penalty
    future_km = km_driven + (km_per_month * months_to_hold)
    cliff_penalty = 0.0

    crossed_50k = km_driven < 50000 <= future_km
    crossed_100k = km_driven < 100000 <= future_km

    if crossed_100k:
        cliff_penalty += 0.045  # 4.5% psychological cliff penalty
    elif crossed_50k:
        cliff_penalty += 0.030  # 3.0% psychological cliff penalty

    # 3. Seasonal Market Demand Adjustment
    season_factor = 0.0
    if season_type == "festive":
        # Diwali / Navratri / Festive demand surge in India
        season_factor = +0.055
    elif season_type == "year_end":
        # December year-end model year changeover discount
        season_factor = -0.035

    net_factor = 1.0 - monthly_deprec - cliff_penalty + season_factor
    projected_price = max(0.0, round(base_price * net_factor, -2))
    delta_inr = round(projected_price - base_price, -2)
    delta_lakh = round(delta_inr / 100000, 2)
    pct_change = round(((projected_price - base_price) / base_price) * 100, 1)

    # 4. Strategic Rationale
    if season_type == "festive" and delta_inr >= 0:
        recommendation = "HOLD FOR FESTIVE PEAK"
        verdict_title = f"🌟 Hold {months_to_hold} Month(s) for Festive Demand Surge"
        rationale = (
            f"Diwali / Navratri consumer buying sentiment creates a +5.5% retail premium, "
            f"offsetting holding depreciation and yielding a net gain of +₹{abs(delta_inr):,.0f} (+{pct_change}%)."
        )
    elif crossed_50k or crossed_100k:
        cliff_km = "1,00,000 km" if crossed_100k else "50,000 km"
        recommendation = "SELL BEFORE MILESTONE CLIFF"
        verdict_title = f"⚠️ Sell Now Before Crossing {cliff_km}!"
        rationale = (
            f"Waiting {months_to_hold} months will push the odometer past {cliff_km} ({future_km:,} km). "
            f"Buyers heavily discount vehicles after crossing this psychological threshold, resulting in a ₹{abs(delta_inr):,.0f} value loss."
        )
    elif delta_inr < 0:
        recommendation = "SELL NOW TO PREVENT LOSS"
        verdict_title = f"📉 Sell Today: Holding Costs ₹{abs(delta_inr):,.0f}"
        rationale = (
            f"Holding the car for {months_to_hold} months adds {km_per_month * months_to_hold:,} km and "
            f"accrues natural depreciation, eroding your sale proceeds by {abs(pct_change)}%."
        )
    else:
        recommendation = "NEUTRAL TIMING"
        verdict_title = f"⚖️ Fair Market Stability (Projected ₹{projected_price/100000:.2f} L)"
        rationale = "Projected depreciation and seasonal demand are roughly balanced across this window."

    return {
        "months_to_hold": months_to_hold,
        "current_price_inr": float(base_price),
        "projected_price_inr": float(projected_price),
        "projected_price_lakh": round(projected_price / 100000, 2),
        "delta_inr": float(delta_inr),
        "delta_lakh": float(delta_lakh),
        "pct_change": float(pct_change),
        "future_odometer": int(future_km),
        "recommendation": recommendation,
        "verdict_title": verdict_title,
        "rationale": rationale,
        "crossed_cliff": bool(crossed_50k or crossed_100k),
    }
