# Agricultural Data Analysis & Climate Impact Dashboard

## Project Title
**Agricultural Data Analysis & Climate Impact Dashboard**

## Overview
This repository provides a Python‑based pipeline that processes satellite imagery and weather observations to assess crop health, detect early heat and drought stress, and visualise risk maps. The workflow combines spectral vegetation indices, temperature anomaly scores, and precipitation deficits to generate a composite stress metric for any agricultural region.

## Core Functionality
- **Crop health tracking** – NDVI (and optional EVI/SAVI) computed from NIR/Red bands with cloud‑mask handling.
- **Anomaly detection** – Land‑Surface Temperature (LST) Z‑scores computed against a rolling 30‑day baseline, and precipitation‑deficit calculations.
- **Integrated stress index** – Weighted combination of NDVI change, temperature Z‑score, and rainfall deficit (the ICSI/CARI score).
- **Interactive visualisation** – Streamlit UI with Plotly charts, Folium maps, and downloadable low‑bandwidth JSON payloads.

## Directory Structure
```
agri_sens_core/
├── .env.example                # Example environment variables (optional)
├── .gitignore
├── .streamlit/
│   └── config.toml             # Streamlit theme & server settings
├── app.py                       # Minimal entry point that forwards to Streamlit app
├── app/                         # Streamlit application package
│   ├── streamlit_app.py         # Main UI definition
│   ├── assets/                  # CSS / static UI assets
│   └── views/                   # Tab modules (analytics, map, export, mission)
├── config/                      # Configuration files (if any)
├── data/                        # Raw / processed raster and CSV data (git‑ignored)
│   ├── raw/.gitkeep
│   ├── interim/.gitkeep
│   └── processed/.gitkeep
├── RESEARCH.md                  # Technical research document (this file)
├── README.md                    # Project documentation (this file)
├── requirements.txt
├── src/                         # Core processing modules
│   ├── analytics/
│   │   ├── risk_engine.py       # Composite Agricultural Risk Index (CARI) implementation
│   │   ├── indices.py           # NDVI/EVI/SAVI calculations
│   │   └── anomalies.py         # Z‑score anomaly detection for soil moisture & LST
│   ├── ingestion/               # NASA POWER API client, STAC client
│   ├── preprocessing/           # Cloud masking, reprojection, temporal compositing
│   └── utils/                   # Logger, custom exceptions
├── tests/                       # Pytest suite covering each module
└── Dockerfile                   # Container for reproducible builds
```

## Architecture Diagram
```
Raw satellite & weather feeds (CSV / GeoTIFF) 
        │
        ▼
+-------------------+   +-------------------+   +-------------------+
| Band Math (NDVI) |   | LST Z‑score       |   | Precip deficit    |
|   - Cloud mask   |   |   (rolling 30d)   |   |   (30‑day sum)    |
+--------+----------+   +--------+----------+   +--------+----------+
         │                     │                      │
         ▼                     ▼                      ▼
          ---------------------------------------------
          | Integrated Crop Stress Index (ICSI / CARI) |
          | 0.40·Φ(‑ΔNDVI·5) + 0.35·Φ(‑Z_SM) + 0.25·Φ(Z_LST) |
          ---------------------------------------------
                              │
                              ▼
                +-------------------------------+
                | GeoPandas GeoDataFrame output |
                +-------------------------------+
                              │
                              ▼
                +-------------------------------+
                | Streamlit Dashboard (maps,   |
                | charts, download JSON)       |
                +-------------------------------+
```

## Tech Stack
- **Language**: Python 3.11
- **Numerical**: NumPy, Pandas
- **Geospatial**: Rasterio, GeoPandas, rioxarray
- **Visualization**: Plotly, Folium & `streamlit-folium`
- **Web UI**: Streamlit
- **Testing**: Pytest
- **Containerisation**: Docker (Python‑slim + GDAL)

## Quick‑Start Guide
```bash
# 1. Clone the repository
git clone https://github.com/Yeabsera-Gezahegn/agri-sens-core.git
cd agri-sens-core

# 2. Create a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. (Optional) Load data – see src/ingestion modules for NASA POWER or STAC usage
#    Example: python -m src.ingestion.climate_api ...

# 5. Run the interactive dashboard
streamlit run app.py
```

The Streamlit app will launch at `http://localhost:8500`. Use the sidebar to select region, date range, and adjust parameters (rainfall rate, slope, distance). The UI displays metric cards for maximum observed rainfall, peak hazard score, and estimated surge arrival time, plus a colour‑coded risk map.

---

*All code and documentation are released under the MIT License.*
