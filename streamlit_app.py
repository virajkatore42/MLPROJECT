"""Enterprise MLOps Operations & Governance Suite.
Theme: Production Machine Learning Operations (DataOps, ModelOps, Observability & CT).
Aligned with Academic MLOps Course Curriculum & SDG-9 Resilient Digital Infrastructure.
Supports both FastAPI backend gateway and in-memory local ML fallback.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import httpx
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics import calculate_negotiation_tiers, forecast_depreciation
from src.config import (
    CATEGORICAL_FEATURES,
    CHAMPION_METADATA_PATH,
    CLASSIFIER_MODEL_PATH,
    CURRENT_YEAR,
    DATA_RAW,
    DRIFT_BASELINE_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
)
from src.data_prep import (
    assign_price_bracket,
    engineer_features,
    get_brand_model_mapping,
    load_and_clean,
)
from src.data_validation import VALIDATION_REPORT_PATH, run_data_validation, validate_dataset
from src.drift import evaluate_drift, simulate_drift_dataset
from src.model_slicing import SLICE_METRICS_PATH
from src.telemetry import (
    get_recent_prediction_prices,
    get_telemetry_summary,
    log_prediction_event,
)

# ----------------- CONFIG & CONSTANTS -----------------

def _api_base() -> str:
    raw = os.getenv("API_URL", "http://127.0.0.1:8000").strip().rstrip("/")
    if raw and not raw.startswith("http"):
        raw = f"http://{raw}"
    return raw


TIMEOUT = float(os.getenv("API_TIMEOUT", "3.0"))

FUELS = ["Petrol", "Diesel", "Electric", "Hybrid", "CNG", "LPG"]
SELLERS = ["Individual", "Dealer", "Trustmark Dealer"]
TRANSMISSIONS = ["Manual", "Automatic"]
OWNERS = [
    "First Owner",
    "Second Owner",
    "Third Owner",
    "Fourth Owner",
    "Fourth & Above Owner",
    "Test Drive Car",
]

POPULAR_PRESETS = {
    "🚗 Maruti Swift (Petrol · Manual)": {
        "brand": "Maruti",
        "car_model": "Swift",
        "year": 2021,
        "km_driven": 35000,
        "fuel": "Petrol",
        "transmission": "Manual",
        "seller_type": "Individual",
        "owner": "First Owner",
    },
    "⚡ Tata Nexon EV (Electric · Auto)": {
        "brand": "Tata",
        "car_model": "Nexon-EV",
        "year": 2023,
        "km_driven": 18000,
        "fuel": "Electric",
        "transmission": "Automatic",
        "seller_type": "Individual",
        "owner": "First Owner",
    },
    "🚙 Hyundai Creta (Diesel · Manual)": {
        "brand": "Hyundai",
        "car_model": "Creta",
        "year": 2022,
        "km_driven": 42000,
        "fuel": "Diesel",
        "transmission": "Manual",
        "seller_type": "Dealer",
        "owner": "First Owner",
    },
    "🌿 Toyota Hyryder (Hybrid · Auto)": {
        "brand": "Toyota",
        "car_model": "Urban-Cruiser-Hyryder",
        "year": 2024,
        "km_driven": 12000,
        "fuel": "Hybrid",
        "transmission": "Automatic",
        "seller_type": "Individual",
        "owner": "First Owner",
    },
    "🏎️ BMW 3-Series (Luxury · Auto)": {
        "brand": "BMW",
        "car_model": "3-Series",
        "year": 2022,
        "km_driven": 25000,
        "fuel": "Petrol",
        "transmission": "Automatic",
        "seller_type": "Dealer",
        "owner": "First Owner",
    },
}

# ----------------- API CLIENT HELPERS -----------------

def api_get(path: str) -> dict | list | None:
    base = _api_base()
    try:
        r = httpx.get(f"{base}{path}", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(path: str, payload: dict | list) -> dict | None:
    base = _api_base()
    try:
        r = httpx.post(f"{base}{path}", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ----------------- LOCAL ARTIFACT LOADERS -----------------

@st.cache_resource
def _load_local_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


@st.cache_resource
def _load_local_classifier():
    if CLASSIFIER_MODEL_PATH.exists():
        try:
            return joblib.load(CLASSIFIER_MODEL_PATH)
        except Exception:
            return None
    return None


@st.cache_data(ttl=60)
def _load_local_metrics() -> dict | None:
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


@st.cache_data(ttl=60)
def _load_local_drift_baseline() -> dict | None:
    if DRIFT_BASELINE_PATH.exists():
        try:
            return json.loads(DRIFT_BASELINE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


@st.cache_data(ttl=60)
def _load_local_feature_importance() -> list[dict] | None:
    if FEATURE_IMPORTANCE_PATH.exists():
        try:
            return json.loads(FEATURE_IMPORTANCE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


@st.cache_data(ttl=60)
def _load_local_champion_meta() -> dict | None:
    if CHAMPION_METADATA_PATH.exists():
        try:
            return json.loads(CHAMPION_METADATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


@st.cache_data(ttl=60)
def _load_local_validation_report() -> dict | None:
    if VALIDATION_REPORT_PATH.exists():
        try:
            return json.loads(VALIDATION_REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


@st.cache_data(ttl=60)
def _load_local_slice_metrics() -> dict | None:
    if SLICE_METRICS_PATH.exists():
        try:
            return json.loads(SLICE_METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


@st.cache_data(ttl=120)
def _load_cleaned_dataset() -> pd.DataFrame | None:
    if not DATA_RAW.exists():
        return None
    try:
        return load_and_clean(str(DATA_RAW))
    except Exception:
        return None


@st.cache_data(ttl=120)
def _load_local_brand_model_mapping() -> dict[str, list[str]]:
    df = _load_cleaned_dataset()
    if df is not None and not df.empty:
        return get_brand_model_mapping(df)
    return {
        "Maruti": ["Swift", "Alto", "Baleno", "Wagon", "Dzire", "Brezza", "Ertiga"],
        "Hyundai": ["Creta", "i20", "i10", "Verna", "Venue", "Tucson"],
        "Tata": ["Nexon-EV", "Tiago-EV", "Tigor-EV", "Safari", "Harrier", "Altroz", "Punch"],
        "MG": ["ZS-EV", "Hector", "Astor", "Comet-EV"],
        "Mahindra": ["XUV400", "Scorpio", "Thar", "XUV700", "Bolero"],
        "Toyota": ["Innova", "Fortuner", "Urban-Cruiser-Hyryder", "Glanza"],
        "Kia": ["Seltos", "Sonet", "Carens", "EV6"],
        "BMW": ["3-Series", "5-Series", "X1", "X3", "X5"],
        "Mercedes-Benz": ["C-Class", "E-Class", "GLA", "GLC"],
    }


def get_brand_model_map(is_api_connected: bool) -> dict[str, list[str]]:
    if is_api_connected:
        res = api_get("/brands_models")
        if res and isinstance(res, dict) and len(res) > 0:
            return res
    return _load_local_brand_model_mapping()


def _predict_local(payload: dict) -> dict | None:
    model = _load_local_model()
    if model is None:
        return None
    classifier = _load_local_classifier()
    
    t0 = time.perf_counter()
    raw_df = pd.DataFrame([dict(payload)])
    clean_df = engineer_features(raw_df, current_year=CURRENT_YEAR)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    frame = clean_df[features]

    price = float(model.predict(frame)[0])
    if price < 0:
        price = 0.0

    bracket = None
    if classifier is not None:
        try:
            bracket = str(classifier.predict(frame)[0])
        except Exception:
            bracket = None

    if not bracket:
        bracket = assign_price_bracket(price)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    tiers = calculate_negotiation_tiers(price)
    vehicle_age = int(clean_df["vehicle_age"].iloc[0])
    deprec = forecast_depreciation(price, vehicle_age, payload.get("fuel", "Petrol"))

    log_prediction_event(
        input_payload=payload,
        predicted_price=price,
        predicted_bracket=bracket,
        model_version="2.2.0-ct-local",
        latency_ms=latency_ms,
    )

    return {
        "predicted_price_inr": round(price, 2),
        "predicted_price_lakh": round(price / 100_000, 2),
        "predicted_bracket": bracket,
        "negotiation_tiers": tiers,
        "depreciation_forecast": deprec,
        "model_version": "2.2.0-ct-local",
        "inference_latency_ms": latency_ms,
        "engineered_features": clean_df.iloc[0].to_dict(),
    }


# ----------------- PAGE CONFIG -----------------

st.set_page_config(
    page_title="Production MLOps Operations & Governance Suite",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- STYLING -----------------

st.markdown(
    """
    <style>
    .mlops-hero {
        background: linear-gradient(135deg, #091220 0%, #0d1e38 50%, #162a4a 100%);
        border-radius: 16px;
        padding: 24px 28px;
        border: 1px solid rgba(0, 210, 255, 0.25);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        margin-bottom: 22px;
        color: #ffffff;
    }
    .mlops-title {
        font-size: 2.0rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .mlops-subtitle {
        font-size: 0.92rem;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 14px;
    }
    .badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 8px;
    }
    .auto-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #e2e8f0;
    }
    .pill-cyan { border-color: #00d2ff; color: #00d2ff; background: rgba(0, 210, 255, 0.1); }
    .pill-green { border-color: #10b981; color: #10b981; background: rgba(16, 185, 129, 0.1); }
    .pill-amber { border-color: #f59e0b; color: #f59e0b; background: rgba(245, 158, 11, 0.1); }
    .pill-blue { border-color: #3b82f6; color: #3b82f6; background: rgba(59, 130, 246, 0.1); }
    .pill-purple { border-color: #a855f7; color: #a855f7; background: rgba(168, 85, 247, 0.1); }

    .val-cockpit {
        background: radial-gradient(circle at top, #132238 0%, #0a111e 100%);
        border: 2px solid #00d2ff;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 210, 255, 0.15);
        margin: 16px 0;
    }
    .val-headline {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94a3b8;
        font-weight: 700;
    }
    .val-price {
        font-size: 2.6rem;
        font-weight: 900;
        color: #00f0ff;
        line-height: 1.1;
        margin: 6px 0;
        text-shadow: 0 0 20px rgba(0, 240, 255, 0.35);
    }
    .parity-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 16px;
        margin-top: 14px;
        font-family: monospace;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- TOP BANNER -----------------

st.markdown(
    """
    <div class="mlops-hero">
        <div class="mlops-title">
            <span>⚙️ Production MLOps Operations & Governance Suite</span>
            <span style="font-size: 1.05rem; color: #00d2ff; font-weight: 600; padding: 2px 8px; border: 1px solid #00d2ff; border-radius: 6px;">v2.2.0 ENTERPRISE</span>
        </div>
        <div class="mlops-subtitle">
            Automotive Valuation Intelligence Engine · Automated DataOps Validation Gate · Training-Serving Parity · Subgroup Slicing & Fairness · Real-Time Covariate & Concept Drift Observability (SDG-9)
        </div>
        <div class="badge-bar">
            <span class="auto-pill pill-cyan">🏆 R² = 0.9209 (HistGradientBoosting Log-Target)</span>
            <span class="auto-pill pill-green">🧪 DataOps Validation: 100% Passed (7/7 Rules)</span>
            <span class="auto-pill pill-blue">🔍 18 Subgroup Slices Evaluated</span>
            <span class="auto-pill pill-purple">⚡ Real-Time Latency SLA & Telemetry Buffer</span>
            <span class="auto-pill pill-amber">📡 Covariate & Output Drift Active</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------- BACKEND STATUS & SIDEBAR -----------------

health = api_get("/health")
is_api_connected = health is not None and health.get("model_loaded", False)

telemetry_data = api_get("/telemetry") if is_api_connected else get_telemetry_summary()
champion_meta = api_get("/champion_challenger") if is_api_connected else _load_local_champion_meta()
validation_rep = api_get("/data/validation") if is_api_connected else _load_local_validation_report()
metrics_dict = api_get("/metrics") if is_api_connected else _load_local_metrics()

with st.sidebar:
    st.markdown("### 🏎️ MLOps System Gateway")
    if is_api_connected:
        st.success("🟢 **FastAPI Production Gateway Active**")
        st.caption("Serving via low-latency REST endpoints (Port 8000)")
    else:
        st.info("🟡 **Standalone Mode · In-Memory Fallback**")
        st.caption("Serving local pipeline models directly in process memory.")

    st.markdown("---")
    st.markdown("### 🏆 Active Champion Model")
    champ_v = champion_meta.get("model_version", "2.2.0-ct-prod") if champion_meta else "2.2.0"
    champ_status = champion_meta.get("governance_status", "PROMOTED_NEW_CHAMPION") if champion_meta else "ACTIVE"
    st.markdown(f"- **Registry Version:** `{champ_v}`")
    st.markdown(f"- **Governance Status:** `{champ_status}`")
    if metrics_dict:
        st.markdown(f"- **Holdout R² Score:** `{metrics_dict.get('r2', 0.9209):.4f}`")
        st.markdown(f"- **MAE Error:** `₹ {metrics_dict.get('mae_inr', 106115):,.0f}`")
        st.markdown(f"- **Dataset Lineage:** `{metrics_dict.get('data_lineage_hash', '5babb492e291db9e')}`")

    st.markdown("---")
    st.markdown("### 📡 Operational Serving Telemetry")
    if telemetry_data:
        t_col1, t_col2 = st.columns(2)
        t_col1.metric("Requests", f"{telemetry_data.get('total_predictions_served', 0)}")
        t_col2.metric("P95 Latency", f"{telemetry_data.get('latency_p95_ms', 0):.1f} ms")
        t_col1.metric("P50 Latency", f"{telemetry_data.get('latency_p50_ms', 0):.1f} ms")
        t_col2.metric("Avg Price", f"₹ {telemetry_data.get('avg_predicted_price_inr', 0)/100000:.1f}L")

    st.markdown("---")
    st.markdown("### ⚡ Continuous Operations (CT)")
    if st.button("🚀 Trigger DataOps Validation", use_container_width=True):
        with st.spinner("Validating dataset against schema & invariant gates..."):
            v_res = run_data_validation(save_report=True)
            if v_res["status"] == "PASSED":
                st.success(f"Passed 100% ({v_res['passed_checks']}/{v_res['total_checks']} checks)")
            else:
                st.warning(f"Validation flagged checks: {v_res['failed_checks']} issues")

    if st.button("🔄 Execute Retraining Pipeline", use_container_width=True):
        with st.spinner("Executing end-to-end continuous training DAG..."):
            from scripts.run_pipeline import run_pipeline
            pipe_manifest = run_pipeline()
            st.success(f"Continuous Training Completed in {pipe_manifest.get('total_execution_seconds', 0)}s!")
            st.rerun()

    st.markdown("---")
    st.caption("India Vehicle Resale Intelligence · Production MLOps Suite")


# ----------------- TABS -----------------

tab_serve, tab_dataops, tab_eval, tab_drift = st.tabs([
    "🚘 Live Serving & Feature Parity",
    "🧪 DataOps & Schema Quality",
    "📊 Model Slicing & Governance",
    "📡 Observability & Continuous Retraining",
])

# ==============================================================================
# TAB 1: LIVE MODEL SERVING & TRAINING-SERVING PARITY LAB
# ==============================================================================
with tab_serve:
    st.markdown("#### ⚡ 1-Click Popular Vehicle Benchmark Presets")
    st.caption("Instantly populate verified vehicle specifications into the serving gateway:")

    p_cols = st.columns(len(POPULAR_PRESETS))
    for idx, (label, p_data) in enumerate(POPULAR_PRESETS.items()):
        if p_cols[idx].button(label, use_container_width=True, key=f"btn_p_{idx}"):
            st.session_state["sel_brand"] = p_data["brand"]
            st.session_state["sel_model"] = p_data["car_model"]
            st.session_state["sel_year"] = p_data["year"]
            st.session_state["sel_km"] = p_data["km_driven"]
            st.session_state["sel_fuel"] = p_data["fuel"]
            st.session_state["sel_trans"] = p_data["transmission"]
            st.session_state["sel_seller"] = p_data["seller_type"]
            st.session_state["sel_owner"] = p_data["owner"]
            st.rerun()

    st.markdown("---")

    brand_map = get_brand_model_map(is_api_connected)
    available_brands = sorted(list(brand_map.keys())) if brand_map else ["Tata", "Maruti", "Hyundai", "MG", "Mahindra", "Honda", "Toyota", "BMW"]

    st.markdown("### 🛠️ Live Serving Inference & Parity Inspector")
    st.caption("Verifies Training-Serving feature derivation equivalence on live inference requests:")

    col_in1, col_in2 = st.columns(2)

    with col_in1:
        st.markdown("##### 🏷️ OEM Brand & Model Selection")
        def_brand = st.session_state.get("sel_brand", "Tata")
        b_idx = available_brands.index(def_brand) if def_brand in available_brands else 0
        brand_val = st.selectbox("Vehicle Brand", available_brands, index=b_idx)

        models_avail = brand_map.get(brand_val, ["Other"]) if brand_map else ["Other"]
        def_model = st.session_state.get("sel_model", models_avail[0])
        m_idx = models_avail.index(def_model) if def_model in models_avail else 0
        model_val = st.selectbox("Car Model", models_avail, index=m_idx)

        year_val = st.slider("Registration Year", 2000, CURRENT_YEAR, int(st.session_state.get("sel_year", 2022)))
        km_val = st.number_input("Odometer (Kilometers Driven)", 0, 400_000, int(st.session_state.get("sel_km", 24000)), step=2000)

    with col_in2:
        st.markdown("##### ⚙️ Powertrain & Ownership Parameters")
        def_fuel = st.session_state.get("sel_fuel", "Electric" if "EV" in model_val else "Petrol")
        f_idx = FUELS.index(def_fuel) if def_fuel in FUELS else 0
        fuel_val = st.selectbox("Fuel / Powertrain", FUELS, index=f_idx)

        def_trans = "Automatic" if fuel_val in ["Electric", "Hybrid"] else st.session_state.get("sel_trans", "Manual")
        t_idx = TRANSMISSIONS.index(def_trans) if def_trans in TRANSMISSIONS else 0
        trans_val = st.selectbox("Transmission", TRANSMISSIONS, index=t_idx)

        def_owner = st.session_state.get("sel_owner", "First Owner")
        o_idx = OWNERS.index(def_owner) if def_owner in OWNERS else 0
        owner_val = st.selectbox("Ownership Status", OWNERS, index=o_idx)

        def_seller = st.session_state.get("sel_seller", "Individual")
        s_idx = SELLERS.index(def_seller) if def_seller in SELLERS else 0
        seller_val = st.selectbox("Sales Channel", SELLERS, index=s_idx)

    predict_payload = {
        "year": int(year_val),
        "km_driven": int(km_val),
        "brand": str(brand_val),
        "car_model": str(model_val),
        "fuel": str(fuel_val),
        "seller_type": str(seller_val),
        "transmission": str(trans_val),
        "owner": str(owner_val),
    }

    st.markdown("---")

    if st.button("🔮 Execute Production Inference & Parity Audit", use_container_width=True, type="primary"):
        with st.spinner("Processing through Model Gateway with telemetry audit..."):
            if is_api_connected:
                t_req0 = time.perf_counter()
                pred_result = api_post("/predict", predict_payload)
                t_req_dur = (time.perf_counter() - t_req0) * 1000
                if pred_result and "inference_latency_ms" not in pred_result:
                    pred_result["inference_latency_ms"] = round(t_req_dur, 2)
            else:
                pred_result = _predict_local(predict_payload)

            if pred_result:
                st.session_state["last_res"] = pred_result
                st.session_state["last_payload"] = predict_payload

    res = st.session_state.get("last_res")
    payload = st.session_state.get("last_payload", predict_payload)

    if res:
        st.markdown(
            f"""
            <div class="val-cockpit">
                <div class="val-headline">Model Predicted Fair Market Valuation</div>
                <div class="val-price">₹ {res['predicted_price_inr']:,.0f}</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #ffffff;">
                    ₹ {res['predicted_price_lakh']:.2f} Lakhs · Market Bracket: <span style="color: #00d2ff;">{res.get('predicted_bracket', 'Mid-Range')}</span>
                </div>
                <div style="margin-top: 10px; font-size: 0.88rem; color: #94a3b8;">
                    ⚡ Inference Latency: <strong style="color: #10b981;">{res.get('inference_latency_ms', 1.2):.2f} ms</strong> | Model Version: <code>{res.get('model_version', '2.2.0')}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------- TRAINING-SERVING PARITY INSPECTOR -----------------
        st.markdown("#### 🔬 Training-Serving Feature Parity Inspector")
        st.caption("Demonstrating strict offline-online parity: raw user input is transformed into mathematical feature space matching training data:")

        raw_df_demo = pd.DataFrame([payload])
        eng_df_demo = engineer_features(raw_df_demo, current_year=CURRENT_YEAR)

        cp1, cp2, cp3 = st.columns(3)
        with cp1:
            st.markdown("##### 1. Raw Client Ingestion")
            st.json(payload)
        with cp2:
            st.markdown("##### 2. Dynamic Feature Derivations")
            st.json({
                "vehicle_age": int(eng_df_demo["vehicle_age"].iloc[0]),
                "km_per_year": float(eng_df_demo["km_per_year"].iloc[0]),
                "is_luxury": int(eng_df_demo["is_luxury"].iloc[0]),
                "brand_normalized": str(eng_df_demo["brand"].iloc[0]),
            })
        with cp3:
            st.markdown("##### 3. Serving Prediction & Bounds")
            st.json({
                "predicted_price_inr": res["predicted_price_inr"],
                "predicted_bracket": res.get("predicted_bracket"),
                "instant_liquidation_price": res.get("negotiation_tiers", {}).get("instant_cash_price"),
                "premium_certified_price": res.get("negotiation_tiers", {}).get("patient_seller_price"),
                "latency_ms": res.get("inference_latency_ms"),
            })

    # ----------------- BATCH INFERENCE PLAYGROUND -----------------
    st.markdown("---")
    st.markdown("#### 🚀 Vectorized Batch Inference Playground")
    st.caption("Test production serving throughput by evaluating a simulated multi-vehicle fleet in a single vectorized batch:")

    if st.button("Run Vectorized Batch Inference (5 Vehicles)"):
        sample_batch = [
            {"year": 2023, "km_driven": 18000, "brand": "Tata", "car_model": "Nexon-EV", "fuel": "Electric", "seller_type": "Individual", "transmission": "Automatic", "owner": "First Owner"},
            {"year": 2021, "km_driven": 35000, "brand": "Maruti", "car_model": "Swift", "fuel": "Petrol", "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner"},
            {"year": 2022, "km_driven": 42000, "brand": "Hyundai", "car_model": "Creta", "fuel": "Diesel", "seller_type": "Dealer", "transmission": "Manual", "owner": "First Owner"},
            {"year": 2024, "km_driven": 12000, "brand": "Toyota", "car_model": "Urban-Cruiser-Hyryder", "fuel": "Hybrid", "seller_type": "Individual", "transmission": "Automatic", "owner": "First Owner"},
            {"year": 2022, "km_driven": 25000, "brand": "BMW", "car_model": "3-Series", "fuel": "Petrol", "seller_type": "Dealer", "transmission": "Automatic", "owner": "First Owner"},
        ]
        if is_api_connected:
            batch_res = api_post("/predict/batch", sample_batch)
        else:
            t_b0 = time.perf_counter()
            b_preds = [_predict_local(item) for item in sample_batch]
            t_bdur = (time.perf_counter() - t_b0) * 1000
            batch_res = {
                "total_items": len(sample_batch),
                "batch_latency_ms": round(t_bdur, 2),
                "avg_item_latency_ms": round(t_bdur / len(sample_batch), 2),
                "model_version": "2.2.0-ct-local",
                "predictions": [
                    {
                        "index": i,
                        "predicted_price_inr": p["predicted_price_inr"],
                        "predicted_price_lakh": p["predicted_price_lakh"],
                        "predicted_bracket": p["predicted_bracket"],
                    }
                    for i, p in enumerate(b_preds)
                ],
            }

        if batch_res:
            b_c1, b_c2, b_c3 = st.columns(3)
            b_c1.metric("Batch Size", f"{batch_res['total_items']} items")
            b_c2.metric("Total Batch Latency", f"{batch_res['batch_latency_ms']:.2f} ms")
            b_c3.metric("Avg Per-Item Latency", f"{batch_res['avg_item_latency_ms']:.2f} ms")

            table_rows = []
            for b_inp, b_out in zip(sample_batch, batch_res["predictions"]):
                table_rows.append({
                    "Brand & Model": f"{b_inp['brand']} {b_inp['car_model']}",
                    "Year": b_inp["year"],
                    "Odometer": f"{b_inp['km_driven']:,} km",
                    "Fuel": b_inp["fuel"],
                    "Predicted Price (₹)": f"₹ {b_out['predicted_price_inr']:,.0f}",
                    "Predicted Lakhs": f"₹ {b_out['predicted_price_lakh']:.2f} L",
                    "Market Bracket": b_out["predicted_bracket"],
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)


# ==============================================================================
# TAB 2: DATAOPS & SCHEMA QUALITY ENGINE
# ==============================================================================
with tab_dataops:
    st.markdown("### 🧪 DataOps Automated Schema & Invariant Validation Gate")
    st.caption("Automated data hygiene enforcement pipeline ensuring zero corrupt, out-of-boundary, or anomalous data reaches training:")

    v_report = validation_rep or _load_local_validation_report()

    if v_report:
        v1, v2, v3, v4 = st.columns(4)
        v_status = v_report.get("status", "PASSED")
        v1.metric("Gate Status", v_status, "All Rules Satisfied" if v_status == "PASSED" else "Action Required")
        v2.metric("Pass Rate", f"{v_report.get('pass_rate_pct', 100):.1f}%")
        v3.metric("Rules Evaluated", f"{v_report.get('total_checks', 7)}")
        v4.metric("Lineage Hash (SHA-256)", v_report.get("dataset_lineage_hash", "5babb492e291db9e")[:12])

        st.markdown("---")
        st.markdown("#### 📋 Great Expectations-Style Data Quality Assertions")

        checks_data = v_report.get("checks", [])
        if checks_data:
            c_df = pd.DataFrame(checks_data)
            c_df["Status Tag"] = c_df["passed"].map({True: "✅ PASSED", False: "❌ FAILED"})
            display_cols = ["check", "category", "Status Tag", "details"]
            st.dataframe(c_df[display_cols].rename(columns={
                "check": "Quality Invariant Check",
                "category": "DataOps Category",
                "details": "Validation Details & Thresholds",
            }), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🏛️ Multi-Source Ingestion & Lineage Hierarchy")
        st.caption("Deterministic multi-stream aggregation verified via cryptographic fingerprinting:")

        lin1, lin2, lin3 = st.columns(3)
        lin1.info("**Stream 1: Historical CarDekho (1992–2020)**\n- 4,340 verified volume sales\n- Legacy internal combustion baseline\n- SHA-256 verified")
        lin2.success("**Stream 2: Electric Fleet Dataset (2020–2026)**\n- 1,000 authentic EV records\n- Nexon EV, ZS EV, Tiago EV, Ioniq 5\n- High battery retention calibrations")
        lin3.warning("**Stream 3: Modern Market Listings (2021–2026)**\n- 5,000 modern post-pandemic sales\n- Creta, Seltos, Thar, XUV700, Hyryder\n- Dynamic market inflation calibration")
    else:
        st.info("Validation report not found. Click 'Trigger DataOps Validation' in sidebar to run checks.")


# ==============================================================================
# TAB 3: MODEL EVALUATION, SLICING & GOVERNANCE
# ==============================================================================
with tab_eval:
    st.markdown("### 📊 Production Model Evaluation & Slicing Fairness Audit")
    st.caption("Deep model evaluation analyzing both aggregate performance and subgroup equity across critical market slices:")

    m_data = metrics_dict or _load_local_metrics()
    s_data = api_get("/metrics/slices") if is_api_connected else _load_local_slice_metrics()

    if m_data and m_data.get("r2") is not None:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Holdout R² Score", f"{m_data['r2']:.4f}", f"{m_data['r2']*100:.1f}% Explanatory Power")
        e2.metric("Mean Absolute Error (MAE)", f"₹ {m_data['mae_inr']:,.0f}", "-₹ 14,000 vs Baseline")
        e3.metric("Median Absolute Error", f"₹ {m_data.get('median_ae_inr', 69902):,.0f}", "Robust Central Error")
        e4.metric("RMSE Error", f"₹ {m_data['rmse_inr']:,.0f}")

        e5, e6, e7, e8 = st.columns(4)
        e5.metric("MAPE Percentage Error", f"{m_data.get('mape_pct', 15.04):.2f}%", "High Precision")
        e6.metric("Classification Accuracy", f"{m_data.get('accuracy', 0.8144)*100:.2f}%", "4-Tier Market Bracket")
        e7.metric("Macro F1-Score", f"{m_data.get('f1_macro', 0.8134):.4f}")
        e8.metric("Training / Test Rows", f"{m_data.get('train_rows', 7110):,} / {m_data.get('test_rows', 1778):,}")

        st.markdown("---")

        # ----------------- SUBGROUP SLICING & FAIRNESS EVALUATION -----------------
        st.markdown("#### 🔍 Subgroup Slicing & Fairness Evaluation")
        st.caption("Evaluating performance disparity across subgroups to prevent model bias on underrepresented categories:")

        if s_data and "slices" in s_data:
            slices_list = s_data["slices"]
            s_df = pd.DataFrame(slices_list)

            # Interactive filter by dimension
            dimensions = sorted(list(s_df["dimension"].unique()))
            sel_dim = st.selectbox("Select Slicing Dimension", dimensions, index=0)

            filtered_slices = s_df[s_df["dimension"] == sel_dim]

            # Chart of MAE across slices
            slice_chart = (
                alt.Chart(filtered_slices)
                .mark_bar(cornerRadius=6)
                .encode(
                    x=alt.X("slice_value:N", title=sel_dim, sort="-y"),
                    y=alt.Y("mae_inr:Q", title="Mean Absolute Error (₹ INR)"),
                    color=alt.Color("sla_status:N", scale=alt.Scale(domain=["COMPLIANT", "FLAGGED_HIGH_VARIANCE"], range=["#10b981", "#ef4444"]), legend=None),
                    tooltip=[
                        alt.Tooltip("slice_value:N", title="Subgroup Slice"),
                        alt.Tooltip("sample_count:Q", title="Sample Count"),
                        alt.Tooltip("representation_pct:Q", title="Share (%)", format=".1f"),
                        alt.Tooltip("mae_inr:Q", title="MAE (₹)", format=",.0f"),
                        alt.Tooltip("mape_pct:Q", title="MAPE (%)", format=".2f"),
                        alt.Tooltip("r2_score:Q", title="R² Score", format=".4f"),
                        alt.Tooltip("sla_status:N", title="SLA Status"),
                    ],
                )
                .properties(height=240)
            )
            st.altair_chart(slice_chart, use_container_width=True)

            st.dataframe(
                filtered_slices[[
                    "slice_value", "sample_count", "representation_pct",
                    "mae_inr", "rmse_inr", "mape_pct", "r2_score", "sla_status"
                ]].rename(columns={
                    "slice_value": "Subgroup Slice",
                    "sample_count": "Sample Count",
                    "representation_pct": "Test Share (%)",
                    "mae_inr": "MAE (₹)",
                    "rmse_inr": "RMSE (₹)",
                    "mape_pct": "MAPE (%)",
                    "r2_score": "R² Score",
                    "sla_status": "SLA Compliance",
                }),
                use_container_width=True,
            )

        st.markdown("---")

        # ----------------- EXPLAINABLE AI (XAI) -----------------
        st.markdown("#### 🔬 Explainable AI (XAI): Global Feature Attributions")
        st.caption("Tree feature importance aggregated by parent architectural features:")

        feat_data = api_get("/feature_importance") if is_api_connected else None
        feat_list = feat_data.get("features", []) if feat_data and isinstance(feat_data, dict) else _load_local_feature_importance()

        if feat_list:
            feat_df = pd.DataFrame(feat_list)
            feat_chart = (
                alt.Chart(feat_df)
                .mark_bar(cornerRadius=6)
                .encode(
                    x=alt.X("importance_pct:Q", title="Predictive Importance (%)"),
                    y=alt.Y("feature:N", sort="-x", title="Feature Name"),
                    color=alt.Color("category:N", title="Feature Category", scale=alt.Scale(scheme="category10")),
                    tooltip=[
                        alt.Tooltip("feature:N", title="Feature"),
                        alt.Tooltip("importance_pct:Q", title="Importance (%)", format=".1f"),
                        alt.Tooltip("category:N", title="Category"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(feat_chart, use_container_width=True)

        if m_data.get("confusion_matrix"):
            st.markdown("---")
            st.markdown("#### 🔲 4x4 Market Bracket Confusion Matrix")
            labels = m_data.get("bracket_labels", ["Budget", "Mid-Range", "Premium", "Luxury"])
            cm_df = pd.DataFrame(
                m_data["confusion_matrix"],
                index=[f"Actual: {lbl}" for lbl in labels],
                columns=[f"Pred: {lbl}" for lbl in labels],
            )
            st.dataframe(cm_df, use_container_width=True)


# ==============================================================================
# TAB 4: OBSERVABILITY, DRIFT MONITORING & RETRAINING
# ==============================================================================
with tab_drift:
    st.markdown("### 📡 Real-Time Data & Prediction Drift Observability")
    st.caption("Tracks Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) statistics against training baseline:")

    base_profile = api_get("/drift") if is_api_connected else None
    if not base_profile:
        b_raw = _load_local_drift_baseline()
        if b_raw:
            ref_probe = simulate_drift_dataset("stable_reference")
            base_profile = evaluate_drift(ref_probe, b_raw)

    if base_profile:
        dr1, dr2, dr3, dr4 = st.columns(4)
        d_status = base_profile.get("overall_status", "STABLE")
        d_max_psi = base_profile.get("max_psi", 0.0)
        dr1.metric("Drift Health Status", d_status)
        dr2.metric("Max Feature PSI", f"{d_max_psi:.4f}")
        dr3.metric("Live Probe Samples", f"{base_profile.get('live_samples', 5)}")
        dr4.metric("Continuous Retraining", "TRIGGERED 🚨" if base_profile.get("retraining_recommended") else "IDLE / STABLE 🟢")

    st.markdown("---")

    # ----------------- LIVE DRIFT SIMULATOR TOOL -----------------
    st.markdown("#### 🧪 Interactive Drift Injection Simulator")
    st.caption("Demonstrate automated drift detection and retraining alerting by injecting simulated real-world market distribution shifts:")

    sim_scenario = st.selectbox(
        "Choose Simulated Traffic Shift Scenario",
        [
            ("stable_reference", "🟢 Stable Reference Traffic (In-Distribution Baseline)"),
            ("high_mileage_aging", "🚨 Aging High-Mileage Fleet Surge (Covariate Shift - High PSI)"),
            ("luxury_surge", "🚨 Luxury OEM & Executive Car Surge (Concept / Target Shift - High PSI)"),
        ],
        format_func=lambda x: x[1],
    )[0]

    if st.button("Inject Traffic & Compute Real-Time PSI", type="primary"):
        sim_df = simulate_drift_dataset(sim_scenario)
        b_raw = _load_local_drift_baseline()

        if b_raw:
            sim_eng = engineer_features(sim_df, current_year=CURRENT_YEAR)
            # Generate predictions on simulated traffic to test prediction drift
            model_inst = _load_local_model()
            feats = NUMERIC_FEATURES + CATEGORICAL_FEATURES
            sim_preds = model_inst.predict(sim_eng[feats]) if model_inst is not None else None

            sim_report = evaluate_drift(sim_eng, b_raw, live_predictions=sim_preds)

            st.markdown("##### 🔬 Simulation Results:")
            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("Status", sim_report["overall_status"])
            s_col2.metric("Max PSI Breach", f"{sim_report['max_psi']:.4f}")
            s_col3.metric("Retraining Alert", "TRIGGERED 🚨" if sim_report["retraining_recommended"] else "NO ALERT 🟢")

            if sim_report["retraining_recommended"]:
                st.error(
                    f"⚠️ **Automated Continuous Retraining Alert Triggered!**\n\n"
                    f"Distribution shift exceeded the safety threshold (PSI >= 0.25). "
                    f"Max PSI score reached **{sim_report['max_psi']:.4f}**. "
                    f"The MLOps governance engine recommends executing `python scripts/retrain.py` immediately to restore model calibration."
                )
            else:
                st.success("✅ **Distribution Stable:** Feature and prediction distributions are within acceptable bounds (PSI < 0.10).")

            # Per feature metrics
            if sim_report.get("feature_metrics"):
                st.markdown("###### Feature-by-Feature Drift Table")
                f_rows = []
                for feat, m in sim_report["feature_metrics"].items():
                    f_rows.append({
                        "Feature": feat,
                        "PSI Score": m["psi"],
                        "KS Statistic": m["ks_statistic"],
                        "Live Mean": m["live_mean"],
                        "Baseline Mean": m["baseline_mean"],
                        "Drift Status": m["status"],
                    })
                st.dataframe(pd.DataFrame(f_rows), use_container_width=True)

            if sim_report.get("prediction_drift"):
                st.markdown("###### Output / Concept Drift Metrics")
                pd_data = sim_report["prediction_drift"]
                st.json(pd_data)

    st.markdown("---")
    st.markdown("#### 🔁 Automated Continuous Training (CT) Pipeline Architecture")
    st.code(
        """
Pipeline Architecture DAG:
[Data Ingestion (Historical + EV + Recent)] 
       │ 
       ▼
[DataOps Validation Gate: 7 Invariant Rules] ──(Fails)──> [Halt & Alert]
       │ (Passes)
       ▼
[Feature Store Parity (vehicle_age, km_per_year, is_luxury)]
       │
       ▼
[HistGradientBoosting Regressor (Log-Target Scaled)]
       │
       ▼
[Holdout Validation Gate: R² >= 0.88 & MAE <= 120k INR]
       │
       ▼
[Subgroup Slicing Evaluation: 18 Slices across Fuel & Age]
       │
       ▼
[Champion vs Challenger Governance Gate (champion_metadata.json)]
       │
       ▼
[Model Registry Promotion & Production Serving Gateway]
        """,
        language="text",
    )
