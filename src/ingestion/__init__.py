"""Data Ingestion Package for AGRI-SENS-CORE.

Provides STAC integration for Sentinel-2 satellite imagery and NASA POWER daily climate data ingestion.
"""

from src.ingestion.climate_api import NASADataPowerIngestionEngine
from src.ingestion.stac_client import SentinelSTACIngestionEngine

__all__ = [
    "SentinelSTACIngestionEngine",
    "NASADataPowerIngestionEngine",
]
