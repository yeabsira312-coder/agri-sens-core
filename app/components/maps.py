"""Folium Map Visualization Components.

Generates interactive Folium maps with AOI bounding boxes, satellite tile layers,
and color-coded agricultural risk pin markers.
"""

from typing import Any, Tuple, Union

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    folium = None
    HAS_FOLIUM = False

# Map risk tier strings to Folium marker colors and hex values

RISK_COLOR_MAP = {
    "NOMINAL": {"icon_color": "green", "hex": "#22c55e"},
    "MODERATE STRESS": {"icon_color": "orange", "hex": "#eab308"},
    "HIGH RISK": {"icon_color": "darkorange", "hex": "#f97316"},
    "CRITICAL RISK": {"icon_color": "red", "hex": "#ef4444"},
}


def create_risk_map(
    center_lat: float,
    center_lon: float,
    bbox: Tuple[float, float, float, float],
    cari_score: float,
    risk_tier: str,
    zoom_start: int = 13,
) -> Any:
    """Create an interactive Folium map centered at AOI coordinates with risk markers.

    Args:
        center_lat: Center latitude.
        center_lon: Center longitude.
        bbox: Bounding box tuple (min_lon, min_lat, max_lon, max_lat).
        cari_score: Computed Composite Agricultural Risk Index score (0-100).
        risk_tier: Risk classification string ("NOMINAL", "MODERATE STRESS", "HIGH RISK", "CRITICAL RISK").
        zoom_start: Initial map zoom level (default: 13).

    Returns:
        folium.Map instance if folium is installed, else None.
    """
    if not HAS_FOLIUM or folium is None:
        return None

    min_lon, min_lat, max_lon, max_lat = bbox

    color_info = RISK_COLOR_MAP.get(risk_tier.upper(), {"icon_color": "blue", "hex": "#3b82f6"})

    # Initialize base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Add Esri World Imagery (Satellite) Tile Layer
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # Draw AOI Bounding Box Rectangle
    bounds = [[min_lat, min_lon], [max_lat, max_lon]]
    folium.Rectangle(
        bounds=bounds,
        color=color_info["hex"],
        weight=2,
        fill=True,
        fill_color=color_info["hex"],
        fill_opacity=0.15,
        popup=f"AOI Bounds: {bbox}",
    ).add_to(m)

    # HTML Popup content for center marker
    popup_html = f"""
    <div style="font-family: Arial, sans-serif; width: 200px; padding: 4px;">
        <h4 style="margin: 0 0 8px 0; color: #0f172a;">AGRI-SENS Field Assessment</h4>
        <p style="margin: 4px 0;"><b>Center:</b> {center_lat:.4f}, {center_lon:.4f}</p>
        <p style="margin: 4px 0;"><b>CARI Risk Score:</b> <span style="font-size: 16px; font-weight: bold; color: {color_info['hex']};">{cari_score}</span></p>
        <p style="margin: 4px 0;"><b>Risk Tier:</b> <span style="font-weight: bold; color: {color_info['hex']};">{risk_tier}</span></p>
    </div>
    """

    # Add Pin Marker
    folium.Marker(
        location=[center_lat, center_lon],
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"CARI Score: {cari_score} ({risk_tier})",
        icon=folium.Icon(color=color_info["icon_color"], icon="leaf", prefix="fa"),
    ).add_to(m)

    # Add Layer Control
    folium.LayerControl().add_to(m)

    return m
