"""Interactive Risk Map Tab View for AGRI-SENS-CORE."""

import folium
import streamlit as st
from streamlit_folium import st_folium

def render_map_tab(center_lat, center_lon, bbox, cari_score, risk_tier, ndvi_obs, z_sm, z_lst):
    st.subheader("🗺️ Interactive Agricultural Risk Map")
    st.write(f"**Current Plot Status:** {risk_tier} | **CARI Score:** {cari_score:.2f}")

    # 1. Base Map centered on location
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=None)

    # 2. Add Base Satellite Imagery
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satellite Imagery',
        overlay=False,
        control=True
    ).add_to(m)

    # 3. Add Boundaries & Place Name Labels Layer (Fixes missing names on zoom)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        attr='Esri Boundaries & Labels',
        name='Place & Road Labels',
        overlay=True,
        control=True
    ).add_to(m)

    # 4. Standard OpenStreetMap Layer (Backup option with names)
    folium.TileLayer('OpenStreetMap', name='Street Map').add_to(m)

    # Highlight Target Plot Bounding Box
    min_lon, min_lat, max_lon, max_lat = bbox
    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        color="#10B981" if risk_tier == "Low Risk" else "#EF4444",
        weight=3,
        fill=True,
        fill_opacity=0.2,
        popup=f"Target Plot Zone ({risk_tier})"
    ).add_to(m)

    # Marker for center
    folium.Marker(
        [center_lat, center_lon],
        popup=f"Center: ({center_lat:.4f}, {center_lon:.4f})\nCARI Score: {cari_score:.2f}",
        icon=folium.Icon(color="green" if risk_tier == "Low Risk" else "red", icon="info-sign")
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # Render Map in Streamlit
    st_folium(m, width="100%", height=500)
