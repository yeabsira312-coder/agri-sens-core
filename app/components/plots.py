from typing import Any
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    go = None
    make_subplots = None
    HAS_PLOTLY = False


def create_ndvi_timeseries_plot(df: pd.DataFrame) -> Any:
    """Generate interactive Plotly figure comparing observed NDVI to historical baseline.

    Args:
        df: pandas.DataFrame indexed by date containing 'NDVI' (or 'NDVI_Observed')
            and optionally 'NDVI_Baseline' and 'NDVI_Std'.

    Returns:
        plotly.graph_objects.Figure instance if plotly is installed, else None.
    """
    if not HAS_PLOTLY or go is None:
        return None

    fig = go.Figure()


    # Determine column names
    ndvi_col = "NDVI" if "NDVI" in df.columns else ("NDVI_Observed" if "NDVI_Observed" in df.columns else df.columns[0])
    baseline_col = "NDVI_Baseline" if "NDVI_Baseline" in df.columns else None
    std_col = "NDVI_Std" if "NDVI_Std" in df.columns else None

    # Dates
    dates = df.index

    # 1. Historical Baseline Envelope (if baseline and std available)
    if baseline_col is not None:
        baseline = df[baseline_col]
        std = df[std_col] if std_col is not None else 0.05
        upper_bound = baseline + std
        lower_bound = baseline - std

        # Upper bound line (invisible line for fill)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=upper_bound,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Lower bound line + shaded confidence ribbon
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=lower_bound,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(16, 185, 129, 0.15)",
                name="10-Yr Historical Baseline Range (±1σ)",
            )
        )

        # Baseline mean line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=baseline,
                mode="lines",
                line=dict(color="#94a3b8", width=2, dash="dash"),
                name="Historical Baseline Mean",
            )
        )

    # 2. Observed NDVI Line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df[ndvi_col],
            mode="lines+markers",
            line=dict(color="#10b981", width=3),
            marker=dict(size=6, color="#059669"),
            name="Observed Sentinel-2 NDVI",
        )
    )

    # Layout styling
    fig.update_layout(
        title="<b>Canopy Health Dynamics (Sentinel-2 NDVI Time Series)</b>",
        xaxis_title="Date",
        yaxis_title="NDVI Value",
        yaxis=dict(range=[-0.1, 1.0], gridcolor="rgba(255, 255, 255, 0.1)"),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#f8fafc", family="Arial, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def create_climate_anomaly_plot(df: pd.DataFrame) -> Any:
    """Generate dual subplot bar chart for Z_SM (Soil Moisture) and Z_LST (Surface Temp) Z-scores.

    Args:
        df: pandas.DataFrame indexed by date containing 'Z_SM' and 'Z_LST'.

    Returns:
        plotly.graph_objects.Figure instance if plotly is installed, else None.
    """
    if not HAS_PLOTLY or make_subplots is None or go is None:
        return None

    fig = make_subplots(

        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "<b>Soil Moisture Anomaly (Z_SM)</b>",
            "<b>Land Surface Temperature Anomaly (Z_LST)</b>",
        ),
    )

    dates = df.index
    z_sm = df["Z_SM"] if "Z_SM" in df.columns else pd.Series(0, index=dates)
    z_lst = df["Z_LST"] if "Z_LST" in df.columns else pd.Series(0, index=dates)

    # 1. Soil Moisture Z-Score Bar Chart
    sm_colors = ["#ef4444" if val < -1.5 else "#3b82f6" for val in z_sm]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=z_sm,
            marker_color=sm_colors,
            name="Z_SM (Soil Moisture)",
        ),
        row=1,
        col=1,
    )

    # Threshold line at Z = -1.5 (Drought)
    fig.add_hline(
        y=-1.5,
        line_dash="dash",
        line_color="#ef4444",
        line_width=2,
        annotation_text="Drought Threshold (-1.5σ)",
        annotation_position="bottom right",
        row=1,
        col=1,
    )

    # 2. Surface Temperature Z-Score Bar Chart
    lst_colors = ["#ef4444" if val > 1.5 else "#f59e0b" for val in z_lst]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=z_lst,
            marker_color=lst_colors,
            name="Z_LST (Surface Temp)",
        ),
        row=2,
        col=1,
    )

    # Threshold line at Z = +1.5 (Heat Stress)
    fig.add_hline(
        y=1.5,
        line_dash="dash",
        line_color="#ef4444",
        line_width=2,
        annotation_text="Heat Stress Threshold (+1.5σ)",
        annotation_position="top right",
        row=2,
        col=1,
    )

    # Layout styling
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#f8fafc", family="Arial, sans-serif"),
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
        height=500,
    )

    fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.1)", row=1, col=1)
    fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.1)", row=2, col=1)
    fig.update_yaxes(title_text="Z-Score", gridcolor="rgba(255, 255, 255, 0.1)", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", gridcolor="rgba(255, 255, 255, 0.1)", row=2, col=1)

    return fig
