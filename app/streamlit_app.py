"""Main Streamlit Application Entrypoint for AGRI-SENS-CORE-V1."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import sys
import pathlib

# Add repository root directory to Python path
sys.path.append(str(pathlib.Path(__file__).parent.parent))
# Configure Streamlit Page
st.set_page_config(
    page_title="AGRI-SENS-CORE-V1",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom CSS if present
css_path = Path(__file__).parent / "assets" / "custom.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Import internal modules
from src.analytics.anomalies import AnomalyDetectionEngine
from src.analytics.indices import VegetationIndexEngine
from src.analytics.risk_engine import RiskEngine
from app.views.tab_analytics import render_analytics_tab
from app.views.tab_export import render_export_tab
from app.views.tab_map import render_map_tab
from app.views.tab_mission import render_mission_tab

# AOI Presets
AOI_PRESETS = {
    "Rift Valley (Nakuru Wheat Zone)": {"lat": -0.30, "lon": 36.07},
    "Machakos Agro-Pastoral Region": {"lat": -1.52, "lon": 37.26},
    "Uasin Gishu Grain Belt": {"lat": 0.52, "lon": 35.28},
    "Custom Location": {"lat": -1.28, "lon": 36.82},
}


@st.cache_data
def run_cached_pipeline(
    lat: float,
    lon: float,
    start_date_str: str,
    end_date_str: str,
    max_cloud: float,
) -> pd.DataFrame:
    """Execute data generation, vegetation indexing, and anomaly detection pipeline.

    Cached via @st.cache_data for fast interactive rendering.
    """
    dates = pd.date_range(start_date_str, end_date_str, freq="D")
    np.random.seed(int(abs(lat * 100 + lon * 10)) % 1000 + 42)

    # Synthetic multi-year climate & optical series
    n = len(dates)
    base_ndvi = 0.5 + 0.2 * np.sin(np.linspace(0, 4 * np.pi, n))
    noise_ndvi = np.random.normal(0, 0.03, n)
    observed_ndvi = np.clip(base_ndvi + noise_ndvi, 0.1, 0.9)

    gwettop = 0.35 + 0.15 * np.cos(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 0.05, n)
    ts_c = 26.0 + 5.0 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 1.5, n)

    df = pd.DataFrame(
        {
            "NDVI": observed_ndvi,
            "NDVI_Baseline": base_ndvi,
            "NDVI_Std": 0.04,
            "GWETTOP": np.clip(gwettop, 0.05, 0.8),
            "TS_C": ts_c,
        },
        index=dates,
    )

    # Calculate weekly climate Z-scores
    df = AnomalyDetectionEngine.calculate_weekly_z_scores(df)

    return df


def main() -> None:
    """Main Streamlit Application Controller."""
    st.sidebar.image(
        "https://raw.githubusercontent.com/feathericons/feather/master/icons/globe.svg",
        width=48,
    )
    st.sidebar.title("AGRI-SENS-CORE Controls")

    # 1. Preset Selector
    preset_choice = st.sidebar.selectbox(
        "Select Location Preset",
        list(AOI_PRESETS.keys()),
        index=0,
    )
    default_coords = AOI_PRESETS[preset_choice]

    # 2. Coordinate Inputs
    center_lat = st.sidebar.number_input(
        "Center Latitude (°N)",
        min_value=-90.0,
        max_value=90.0,
        value=default_coords["lat"],
        step=0.01,
        format="%.4f",
    )
    center_lon = st.sidebar.number_input(
        "Center Longitude (°E)",
        min_value=-180.0,
        max_value=180.0,
        value=default_coords["lon"],
        step=0.01,
        format="%.4f",
    )

    # Calculate 0.05 degree bounding box buffer (~5 km box)
    buf = 0.05
    bbox = (
        round(center_lon - buf, 4),
        round(center_lat - buf, 4),
        round(center_lon + buf, 4),
        round(center_lat + buf, 4),
    )

    st.sidebar.markdown("---")

    # 3. Date & Cloud Cover Controls
    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1:
        start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
    with col_d2:
        end_date = st.date_input("End Date", pd.to_datetime("2023-06-30"))

    max_cloud = st.sidebar.slider(
        "Max Cloud Cover Threshold (%)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
    )

    # Execute cached pipeline
    df = run_cached_pipeline(
        lat=center_lat,
        lon=center_lon,
        start_date_str=str(start_date),
        end_date_str=str(end_date),
        max_cloud=max_cloud,
    )

    # Latest observation parameters for Risk Engine
    latest_row = df.iloc[-1]
    ndvi_obs = float(latest_row.get("NDVI", 0.6))
    ndvi_base = float(latest_row.get("NDVI_Baseline", 0.6))
    z_sm = float(latest_row.get("Z_SM", 0.0))
    z_lst = float(latest_row.get("Z_LST", 0.0))

    # Compute CARI Score & Risk Tier
    cari_score, risk_tier = RiskEngine.compute_cari(
        ndvi_obs=ndvi_obs,
        ndvi_baseline=ndvi_base,
        z_sm=z_sm,
        z_lst=z_lst,
    )

    # Render Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. Mission Narrative",
            "2. Interactive Risk Map",
            "3. Analytics & Time Series",
            "4. Data Export Hub",
        ]
    )

    with tab1:
        render_mission_tab()

    with tab2:
        render_map_tab(
            center_lat=center_lat,
            center_lon=center_lon,
            bbox=bbox,
            cari_score=cari_score,
            risk_tier=risk_tier,
            ndvi_obs=ndvi_obs,
            z_sm=z_sm,
            z_lst=z_lst,
        )

    with tab3:
        render_analytics_tab(df)

    with tab4:
        render_export_tab(
            df=df,
            center_lat=center_lat,
            center_lon=center_lon,
            bbox=bbox,
            cari_score=cari_score,
            risk_tier=risk_tier,
            ndvi_obs=ndvi_obs,
            z_sm=z_sm,
            z_lst=z_lst,
        )


if __name__ == "__main__":
    main()
