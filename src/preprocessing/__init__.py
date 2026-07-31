"""Spatial Data Preprocessing Package for AGRI-SENS-CORE.

Includes SCL cloud masking, spatial CRS grid reprojection, and 10-day median temporal compositing.
"""

from src.preprocessing.cloud_masking import mask_sentinel2_clouds
from src.preprocessing.reproject import align_spatial_grid
from src.preprocessing.temporal import create_10day_median_composites

__all__ = [
    "mask_sentinel2_clouds",
    "align_spatial_grid",
    "create_10day_median_composites",
]
