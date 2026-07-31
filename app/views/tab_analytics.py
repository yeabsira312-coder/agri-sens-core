"""Tab View 3: Analytics & Time Series Engine."""

import pandas as pd
import streamlit as st

from app.components.plots import (
    create_climate_anomaly_plot,
    create_ndvi_timeseries_plot,
)


def render_analytics_tab(df: pd.DataFrame) -> None:
    """Render Analytics & Time Series view.

    Args:
        df: Input pandas.DataFrame containing time-series observations,
            NDVI, baseline values, Z_SM, Z_LST, and risk flags.
    """
    st.subheader("📈 Time-Series Analytics & Anomaly Diagnostics")

    # 1. NDVI Time Series Plot
    st.plotly_chart(create_ndvi_timeseries_plot(df), use_container_width=True)

    st.markdown("---")

    # 2. Climate Z-Score Anomaly Bar Charts
    st.plotly_chart(create_climate_anomaly_plot(df), use_container_width=True)

    st.markdown("---")

    st.subheader("📋 Climate Anomaly & Vegetation Data Table")

    # Format table columns for clean display
    display_df = df.copy()
    if isinstance(display_df.index, pd.DatetimeIndex):
        display_df.index = display_df.index.strftime("%Y-%m-%d")

    # Select key numerical columns
    cols = [c for c in display_df.columns if c in ["NDVI", "NDVI_Baseline", "TS_C", "GWETTOP", "Z_SM", "Z_LST", "Drought_Risk", "Heat_Stress_Risk"]]
    if cols:
        st.dataframe(
            display_df[cols].style.highlight_max(axis=0, color="rgba(239, 68, 68, 0.2)"),
            use_container_width=True,
        )
    else:
        st.dataframe(display_df, use_container_width=True)
