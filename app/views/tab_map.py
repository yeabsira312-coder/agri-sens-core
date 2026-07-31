"""Tab View 2: Interactive Risk Map & Spatial Assessment."""

from typing import Tuple
import streamlit as st

from app.components.maps import create_risk_map

# Optional streamlit_folium import
try:
    from streamlit_folium import st_folium
    HAS_STREAMLIT_FOLIUM = True
except ImportError:
    st_folium = None
    HAS_STREAMLIT_FOLIUM = False


def render_map_tab(
    center_lat: float,
    center_lon: float,
    bbox: Tuple[float, float, float, float],
    cari_score: float,
    risk_tier: str,
    ndvi_obs: float,
    z_sm: float,
    z_lst: float,
) -> None:
    """Render Interactive Risk Map & Spatial Assessment view.

    Args:
        center_lat: Center latitude.
        center_lon: Center longitude.
        bbox: Bounding box tuple.
        cari_score: CARI risk score (0-100).
        risk_tier: Risk classification string.
        ndvi_obs: Observed NDVI value.
        z_sm: Soil moisture Z-score.
        z_lst: Land surface temperature Z-score.
    """
    st.subheader("🗺️ Interactive Field Risk Map & Spatial AOI Boundary")

    col_map, col_info = st.columns([3, 2])

    with col_map:
        # Create Folium Map instance
        folium_map = create_risk_map(
            center_lat=center_lat,
            center_lon=center_lon,
            bbox=bbox,
            cari_score=cari_score,
            risk_tier=risk_tier,
        )

        if HAS_STREAMLIT_FOLIUM:
            st_folium(folium_map, width=700, height=480)
        else:
            # Fallback HTML rendering
            map_html = folium_map._repr_html_()
            st.components.v1.html(map_html, height=480)

    with col_info:
        st.markdown("### 📊 Field Risk Summary")


        # Badge color class
        badge_cls = {
            "NOMINAL": "badge-nominal",
            "MODERATE STRESS": "badge-moderate",
            "HIGH RISK": "badge-high",
            "CRITICAL RISK": "badge-critical",
        }.get(risk_tier.upper(), "badge-nominal")

        st.markdown(
            f"""
            <div style="background: rgba(30,41,59,0.7); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
                <p style="margin: 0 0 6px 0; color: #94a3b8;">Current Status:</p>
                <span class="{badge_cls}" style="font-size: 1.1rem;">{risk_tier}</span>
                <h2 style="color: #f8fafc; margin: 12px 0 4px 0;">CARI Index: {cari_score} / 100</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.write(f"**Target Center:** Lat `{center_lat:.4f}`, Lon `{center_lon:.4f}`")
        st.write(f"**Bounding Box Extent:** `[{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]`")

        st.markdown("---")
        st.markdown("**Sub-Index Breakdown:**")
        st.metric(label="Observed Canopy NDVI", value=f"{ndvi_obs:.3f}")
        st.metric(label="Soil Moisture Deficit (Z_SM)", value=f"{z_sm:+.2f} σ")
        st.metric(label="Surface Temp Anomaly (Z_LST)", value=f"{z_lst:+.2f} σ")
