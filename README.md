# AgriSens Core — Agricultural Remote Sensing & Thermal Anomaly Pipeline

## Overview
AgriSens Core is an open‑source pipeline for processing satellite imagery to assess crop canopy health, compute vegetation indices such as NDVI, and detect localized temperature and precipitation anomalies. By transforming raw spectral bands into actionable risk metrics, the system flags potential stress weeks before visual damage appears, supporting early‑season decision making for agronomists and researchers.

## Architecture Diagram
```
Raw Inputs (NIR, Red, Thermal IR, Precip)  
        │
        ▼
+----------------------+   +--------------------+   +-------------------+
| Band Math (NDVI)    |   | Thermal Anomaly    |   | Precip Deficit    |
|  - Cloud mask       |   |  Z‑score vs hist   |   |  Indexing         |
+----------+-----------+   +----------+---------+   +---------+---------+
           │                       │                       │
           ▼                       ▼                       ▼
        +-----------------------------------------------+
        | Integrated Crop Stress Index (ICSI)           |
        |  Combines NDVI, Temp Z‑score, Rain deficit    |
        +----------------------+------------------------+
                               │
                               ▼
                     +-------------------+
                     | Risk Zoning Map   |
                     | (GeoDataFrame)    |
                     +-------------------+
                               │
                               ▼
                     +-------------------+
                     | Streamlit Dashboard |
                     +-------------------+
```

## Key Capabilities
- **NDVI band math & cloud‑mask filtering** – Accurate vegetation health metrics from near‑infrared and red bands.
- **Thermal anomaly detection** – Statistical Z‑score analysis of Land Surface Temperature (LST) against historical baselines.
- **Precipitation Deficit Indexing** – Quantifies rainfall shortfalls across critical growth windows.
- **Spatial risk mapping & modular DataFrame export** – Generates GeoPandas `GeoDataFrame`s for downstream GIS work.

## Tech Stack
- Python 3
- Pandas
- NumPy
- SciPy
- Rasterio
- GeoPandas
- Matplotlib
- Streamlit

## Directory Tree
```
agri_sens_core/
├── data/                # Raw and processed raster/CSV files (not version‑controlled)
├── src/                 # Source code
│   ├── __init__.py
│   ├── data_loader.py
│   ├── ndvi_processor.py
│   ├── thermal_anomaly.py
│   └── risk_index.py
├── app.py               # Streamlit entry point
├── RESEARCH.md
├── README.md            # *(this file)*
├── requirements.txt
└── .gitignore
```

## Quick Start
```bash
# Clone the repository
git clone https://github.com/Yeabsera-Gezahegn/agri-sens-core.git
cd agri-sens-core

# Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the data loader (example)
python src/data_loader.py --input data/raw/

# Launch the interactive dashboard
streamlit run app.py
```

For detailed usage, see the docstrings in each `src/` module and the Streamlit UI.
