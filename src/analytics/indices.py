"""Vegetation Indices Engine Module.

Provides vectorized calculation of spectral vegetation indices (NDVI, EVI, SAVI)
with zero-division protection, array shape validation, and range bound clipping.
"""

from typing import Any, Optional, Union
import numpy as np


from src.utils.exceptions import InvalidIndexCalculationError
from src.utils.logger import get_logger

logger = get_logger("agri_sens.analytics.indices")

ArrayLike = Union[float, int, np.ndarray, Any]


class VegetationIndexEngine:
    """Engine for computing scientific vegetation indices from spectral bands."""

    @staticmethod
    def _validate_inputs(
        nir: ArrayLike,
        red: ArrayLike,
        blue: Optional[ArrayLike] = None,
    ) -> None:
        """Validate input band array shapes and types.

        Raises:
            InvalidIndexCalculationError: If input shapes mismatch or inputs are non-numeric.
        """
        try:
            nir_arr = np.asarray(nir)
            red_arr = np.asarray(red)

            if nir_arr.dtype.kind not in ("f", "i", "u") or red_arr.dtype.kind not in ("f", "i", "u"):
                raise InvalidIndexCalculationError(
                    message="Band array inputs must contain numeric data types.",
                    details={"nir_type": str(type(nir)), "red_type": str(type(red))},
                )

            if nir_arr.shape != red_arr.shape:
                raise InvalidIndexCalculationError(
                    message=f"Band array shape mismatch: NIR shape {nir_arr.shape} != RED shape {red_arr.shape}",
                    details={"nir_shape": nir_arr.shape, "red_shape": red_arr.shape},
                )

            if blue is not None:
                blue_arr = np.asarray(blue)
                if blue_arr.dtype.kind not in ("f", "i", "u"):
                    raise InvalidIndexCalculationError("BLUE band input must contain numeric data type.")
                if blue_arr.shape != nir_arr.shape:
                    raise InvalidIndexCalculationError(
                        message=f"BLUE band shape {blue_arr.shape} does not match NIR shape {nir_arr.shape}",
                    )

        except InvalidIndexCalculationError:
            raise
        except Exception as err:
            raise InvalidIndexCalculationError(
                message="Failed to validate spectral band input arrays.",
                details={"error": str(err)},
            ) from err

    @classmethod
    def compute_ndvi(cls, nir: ArrayLike, red: ArrayLike) -> ArrayLike:
        """Compute Normalized Difference Vegetation Index (NDVI).

        NDVI = (NIR - RED) / (NIR + RED)

        Args:
            nir: Near-Infrared band array or scalar.
            red: Red band array or scalar.

        Returns:
            NDVI array or scalar bounded in range [-1.0, 1.0].

        Raises:
            InvalidIndexCalculationError: On input shape mismatch or invalid types.
        """
        cls._validate_inputs(nir, red)
        logger.debug("Computing NDVI index...")

        try:
            nir_arr = np.asarray(nir, dtype=np.float64)
            red_arr = np.asarray(red, dtype=np.float64)

            with np.errstate(divide="ignore", invalid="ignore"):
                denom = nir_arr + red_arr
                numer = nir_arr - red_arr
                ndvi = np.where(denom == 0.0, np.nan, numer / denom)
                ndvi = np.clip(ndvi, -1.0, 1.0)

            if np.isscalar(nir) and np.isscalar(red):
                return float(np.asarray(ndvi).item())
            return ndvi

        except Exception as err:
            logger.error(f"Error computing NDVI: {err}")
            raise InvalidIndexCalculationError(
                message="NDVI index calculation failed.",
                details={"error": str(err)},
            ) from err

    @classmethod
    def compute_evi(cls, nir: ArrayLike, red: ArrayLike, blue: ArrayLike) -> ArrayLike:
        """Compute Enhanced Vegetation Index (EVI).

        EVI = 2.5 * (NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0)

        Args:
            nir: Near-Infrared band array or scalar.
            red: Red band array or scalar.
            blue: Blue band array or scalar.

        Returns:
            EVI array or scalar.

        Raises:
            InvalidIndexCalculationError: On input shape mismatch or invalid types.
        """
        cls._validate_inputs(nir, red, blue)
        logger.debug("Computing EVI index...")

        try:
            nir_arr = np.asarray(nir, dtype=np.float64)
            red_arr = np.asarray(red, dtype=np.float64)
            blue_arr = np.asarray(blue, dtype=np.float64)

            with np.errstate(divide="ignore", invalid="ignore"):
                denom = nir_arr + 6.0 * red_arr - 7.5 * blue_arr + 1.0
                numer = 2.5 * (nir_arr - red_arr)
                evi = np.where(denom == 0.0, np.nan, numer / denom)

            if np.isscalar(nir) and np.isscalar(red) and np.isscalar(blue):
                return float(np.asarray(evi).item())
            return evi

        except Exception as err:
            logger.error(f"Error computing EVI: {err}")
            raise InvalidIndexCalculationError(
                message="EVI index calculation failed.",
                details={"error": str(err)},
            ) from err

    @classmethod
    def compute_savi(cls, nir: ArrayLike, red: ArrayLike, l_factor: float = 0.5) -> ArrayLike:
        """Compute Soil-Adjusted Vegetation Index (SAVI).

        SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)

        Args:
            nir: Near-Infrared band array or scalar.
            red: Red band array or scalar.
            l_factor: Soil brightness correction factor L (default: 0.5).

        Returns:
            SAVI array or scalar bounded in range [-1.0, 1.0].

        Raises:
            InvalidIndexCalculationError: On input shape mismatch or invalid types.
        """
        cls._validate_inputs(nir, red)
        logger.debug(f"Computing SAVI index with L={l_factor}...")

        try:
            nir_arr = np.asarray(nir, dtype=np.float64)
            red_arr = np.asarray(red, dtype=np.float64)

            with np.errstate(divide="ignore", invalid="ignore"):
                denom = nir_arr + red_arr + l_factor
                numer = (nir_arr - red_arr) * (1.0 + l_factor)
                savi = np.where(denom == 0.0, np.nan, numer / denom)
                savi = np.clip(savi, -1.0, 1.0)

            if np.isscalar(nir) and np.isscalar(red):
                return float(np.asarray(savi).item())
            return savi

        except Exception as err:
            logger.error(f"Error computing SAVI: {err}")
            raise InvalidIndexCalculationError(
                message="SAVI index calculation failed.",
                details={"error": str(err)},
            ) from err

