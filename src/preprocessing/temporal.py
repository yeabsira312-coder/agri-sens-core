"""Temporal Compositing Module.

Provides temporal resampling and median aggregation methods for multi-temporal
raster time series.
"""

from typing import Any
import numpy as np

from src.utils.exceptions import SpatialAlignmentError
from src.utils.logger import get_logger

logger = get_logger("agri_sens.preprocessing.temporal")


def create_10day_median_composites(
    dataset: Any,
    freq: str = "10D",
    time_dim: str = "time",
) -> Any:
    """Resample irregular time series raster observations into 10-day median composites.

    Computes pixel-wise median aggregations skipping NaN values across each 10-day window
    to eliminate cloud artifacts and build seamless temporal composites.

    Args:
        dataset: xarray.Dataset with a datetime 'time' dimension.
        freq: Frequency string for resampling window (default: '10D' for 10-day composites).
        time_dim: Name of the temporal dimension coordinate (default: 'time').

    Returns:
        xarray.Dataset containing 10-day median composite rasters.

    Raises:
        SpatialAlignmentError: If dataset lacks the specified time dimension or resampling fails.
    """
    logger.info(f"Creating temporal median composites with frequency '{freq}' along dimension '{time_dim}'")

    if not hasattr(dataset, "dims") or time_dim not in dataset.dims:
        logger.error(f"Dataset lacks required temporal dimension '{time_dim}'. Found dims: {list(getattr(dataset, 'dims', []))}")
        raise SpatialAlignmentError(
            message=f"Dataset is missing required temporal dimension '{time_dim}'.",
            details={"available_dims": list(getattr(dataset, "dims", []))},
        )

    try:
        # Perform 10-day resampling with skipna=True median aggregation
        resampled = dataset.resample({time_dim: freq})
        composite_ds = resampled.median(skipna=True)

        logger.info(f"Successfully generated 10-day median composites. New time steps: {len(composite_ds[time_dim])}")
        return composite_ds

    except Exception as err:
        logger.error(f"Failed to compute temporal median composites: {err}")
        raise SpatialAlignmentError(
            message=f"Failed to generate temporal median composites for frequency '{freq}'.",
            details={"freq": freq, "error": str(err)},
        ) from err
