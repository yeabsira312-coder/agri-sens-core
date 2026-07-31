"""Tab View 1: Mission Narrative & System Architecture."""

import streamlit as st


def render_mission_tab() -> None:
    """Render Mission Narrative & System Architecture view."""
    st.markdown(
        """
        <div class="hero-container">
            <h1 style="color: #10b981; margin-bottom: 8px;">🌾 AGRI-SENS-CORE-V1</h1>
            <h3 style="color: #f8fafc; margin-top: 0;">Multi-Source Satellite & Climate Risk Engine</h3>
            <p style="color: #94a3b8; font-size: 1.05rem;">
                AGRI-SENS-CORE integrates high-resolution Sentinel-2 L2A optical imagery from Microsoft Planetary Computer
                with multi-year NASA POWER Agroclimatology data. By modeling canopy drop (NDVI), soil moisture deficits (Z_SM),
                and thermal stress (Z_LST), the system provides early warning intelligence for climate risk mitigation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Performance / System Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Satellite Resolution",
            value="10 Meters",
            delta="Sentinel-2 L2A",
        )

    with col2:
        st.metric(
            label="Temporal Resampling",
            value="10-Day Median",
            delta="NaN-Skipping Composite",
        )

    with col3:
        st.metric(
            label="Climate Baseline",
            value="10-Year Rolling",
            delta="NASA POWER Daily",
        )

    with col4:
        st.metric(
            label="Risk Model",
            value="CARI Score",
            delta="0 - 100 Index",
        )

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📌 System Architecture & Pipeline Workflow")
        st.markdown(
            r"""
            1. **Data Ingestion Engine (`src/ingestion/`)**:
               - **SentinelSTACIngestionEngine**: Queries Microsoft Planetary Computer STAC catalog for `sentinel-2-l2a` scenes, signs asset links via SAS tokens, and streams spectral bands (`B02`, `B03`, `B04`, `B08`, `SCL`).
               - **NASADataPowerIngestionEngine**: Queries NASA POWER REST API for daily Land Surface Temperature (`TS_C`) and Topsoil Moisture (`GWETTOP`).

            2. **Preprocessing Pipeline (`src/preprocessing/`)**:
               - **SCL Cloud Masking**: Filters Scene Classification Layer (retaining Vegetation, Soil, Water, Unclassified; masking clouds/shadows to `NaN`).
               - **Spatial Reprojection**: Standardizes coordinate grids to `EPSG:4326` using `rioxarray` Bilinear and Nearest-Neighbor interpolation.
               - **10-Day Compositing**: Resamples time series into 10-day `NaN`-skipping median composites.

            3. **Scientific Feature Engineering (`src/analytics/`)**:
               - **Vegetation Indices**: Computes NDVI, EVI, and SAVI with zero-division protection and range clipping.
               - **Weekly Z-Score Anomaly Detection**: Calculates calendar-week standardized Z-scores ($Z_{\text{SM}}$, $Z_{\text{LST}}$) to flag Drought ($Z_{\text{SM}} < -1.5$) and Heat Stress ($Z_{\text{LST}} > +1.5$).
               - **Composite Agricultural Risk Index (CARI)**: Integrates canopy drop ($\Delta\text{NDVI}$), soil moisture deficit, and thermal stress into a unified 4-tier risk classification (`NOMINAL`, `MODERATE STRESS`, `HIGH RISK`, `CRITICAL RISK`).
            """

        )

    with col_right:
        st.subheader("📐 CARI Risk Formulation")
        st.latex(
            r"\text{CARI} = \left[ 0.40 \cdot \Phi(-\Delta\text{NDVI} \cdot 5) + 0.35 \cdot \Phi(-Z_{\text{SM}}) + 0.25 \cdot \Phi(Z_{\text{LST}}) \right] \times 100"
        )
        st.markdown(
            """
            **Risk Tier Classifications:**
            - 🟢 **NOMINAL (<30)**: Optimal crop health and moisture conditions.
            - 🟡 **MODERATE STRESS (30-55)**: Early warning indicator; monitor soil moisture.
            - 🟠 **HIGH RISK (55-75)**: Significant canopy decline or moisture deficit.
            - 🔴 **CRITICAL RISK (>=75)**: Severe compound heat & drought stress.
            """
        )
