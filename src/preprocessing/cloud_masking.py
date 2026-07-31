"""Sentinel-2 Cloud Masking Module.

Applies Scene Classification Layer (SCL) filtering to mask cloud, shadow,
and defective pixels with NaN values across spectral bands.
"""

from typing import Any, Sequence, Tuple, Union
import numpy as np

from src.utils.exceptions import SpatialAlignmentError
from src.utils.logger import get_logger

logger = get_logger("agri_sens.preprocessing.cloud_masking")

# Valid Sentinel-2 L2A SCL pixel classes to retain
# 4: Vegetation, 5: Not-vegetated, 6: Water, 7: Unclassified
DEFAULT_VALID_SCL_CLASSES = (4, 5, 6, 7)

# Invalid SCL pixel classes to mask out:
# 0: No data, 1: Saturated/Defective, 2: Dark area pixels, 3: Cloud shadows,
# 8: Cloud medium probability, 9: Cloud high probability, 10: Thin cirrus, 11: Snow/Ice
DEFAULT_INVALID_SCL_CLASSES = (0, 1, 2, 3, 8, 9, 10, 11)


def mask_sentinel2_clouds(
    dataset: Any,
    scl_band: str = "SCL",
    valid_classes: Sequence[int] = DEFAULT_VALID_SCL_CLASSES,
) -> Any:
    """Mask cloud and invalid pixels in a Sentinel-2 xarray Dataset using the SCL band.

    Args:
        dataset: xarray.Dataset containing spectral bands and SCL layer.
        scl_band: Name of the Scene Classification Layer band variable (default: 'SCL').
        valid_classes: Sequence of integer SCL class IDs to retain as valid observations.

    Returns:
        xarray.Dataset with invalid/cloud pixels converted to NaN across spectral variables.

    Raises:
        SpatialAlignmentError: If dataset lacks the specified SCL band.
    """
    logger.info(f"Applying Sentinel-2 SCL cloud masking (scl_band='{scl_band}', valid_classes={list(valid_classes)})")

    # Check if dataset has data_vars / is xarray Dataset
    if hasattr(dataset, "data_vars"):
        # Case-insensitive SCL band lookup
        matching_band = None
        for var in dataset.data_vars:
            if str(var).upper() == scl_band.upper():
                matching_band = var
                break

        if matching_band is None:
            logger.error(f"SCL band '{scl_band}' not found in dataset variables: {list(dataset.data_vars)}")
            raise SpatialAlignmentError(
                message=f"Dataset is missing required Scene Classification Layer (SCL) band '{scl_band}'.",
                details={"available_bands": [str(v) for v in dataset.data_vars]},
            )

        scl_array = dataset[matching_band]
        is_valid = scl_array.isin(list(valid_classes))

        # Apply valid mask across spectral data variables
        masked_ds = dataset.where(is_valid)

        # Preserve integer SCL layer values if needed or retain masked dataset
        logger.info("Cloud masking successfully applied across dataset spectral variables.")
        return masked_ds

    else:
        logger.error("Input dataset object does not possess 'data_vars' attribute.")
        raise SpatialAlignmentError(message="Invalid dataset object provided for cloud masking.")
