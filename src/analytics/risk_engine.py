"""Composite Agricultural Risk Index (CARI) Engine Module.

Computes multi-source agricultural risk scores (0-100) combining canopy loss (NDVI drop),
soil moisture deficits (Z_SM), and land surface heat stress (Z_LST).
"""

import math
from typing import Tuple, Union
import numpy as np

from src.utils.exceptions import InvalidIndexCalculationError
from src.utils.logger import get_logger

logger = get_logger("agri_sens.analytics.risk_engine")


class RiskEngine:
    """Engine for computing Composite Agricultural Risk Index (CARI) and stress classifications."""

    @staticmethod
    def logistic_transform(z: float) -> float:
        """Standard logistic sigmoid transformation function.

        Phi(z) = 1 / (1 + exp(-z))

        Args:
            z: Input numerical value.

        Returns:
            Transformed value bounded strictly in range (0.0, 1.0).
        """
        # Clamp z to avoid overflow in exp
        z_clamped = max(-50.0, min(50.0, float(z)))
        return 1.0 / (1.0 + math.exp(-z_clamped))

    @classmethod
    def compute_cari(
        cls,
        ndvi_obs: float,
        ndvi_baseline: float,
        z_sm: float,
        z_lst: float,
    ) -> Tuple[float, str]:
        """Calculate Composite Agricultural Risk Index (CARI) and assign risk classification tier.

        CARI = [0.40 * Phi(-delta_NDVI * 5) + 0.35 * Phi(-Z_SM) + 0.25 * Phi(Z_LST)] * 100

        Risk Classification Tiers:
        - CARI < 30.0: "NOMINAL"
        - 30.0 <= CARI < 55.0: "MODERATE STRESS"
        - 55.0 <= CARI < 75.0: "HIGH RISK"
        - CARI >= 75.0: "CRITICAL RISK"

        Args:
            ndvi_obs: Observed NDVI value.
            ndvi_baseline: Historical baseline/expected NDVI value.
            z_sm: Soil Moisture Z-score (Z_SM).
            z_lst: Land Surface Temperature Z-score (Z_LST).

        Returns:
            Tuple of (cari_score: float, risk_classification: str).

        Raises:
            InvalidIndexCalculationError: If any input value is non-numeric or NaN.
        """
        try:
            # Check non-NaN / numeric
            inputs = [ndvi_obs, ndvi_baseline, z_sm, z_lst]
            if any(math.isnan(float(x)) if isinstance(x, (int, float)) else True for x in inputs):
                raise InvalidIndexCalculationError("CARI computation inputs must be valid non-NaN numeric values.")
        except (ValueError, TypeError) as err:
            raise InvalidIndexCalculationError(
                message="Non-numeric input encountered during CARI risk calculation.",
                details={"error": str(err)},
            ) from err

        try:
            # Calculate relative NDVI change: delta_NDVI = (ndvi_obs - ndvi_baseline) / ndvi_baseline
            if ndvi_baseline != 0.0:
                delta_ndvi = (ndvi_obs - ndvi_baseline) / ndvi_baseline
            else:
                delta_ndvi = 0.0

            # Logistic risk components
            # 1. Canopy loss risk: Phi(-delta_NDVI * 5)
            canopy_risk = cls.logistic_transform(-delta_ndvi * 5.0)

            # 2. Soil moisture deficit risk: Phi(-Z_SM)
            moisture_risk = cls.logistic_transform(-z_sm)

            # 3. Heat stress risk: Phi(Z_LST)
            heat_risk = cls.logistic_transform(z_lst)

            # Weighted CARI score (0 - 100)
            cari_score = (0.40 * canopy_risk + 0.35 * moisture_risk + 0.25 * heat_risk) * 100.0
            cari_score = round(float(cari_score), 2)

            # Assign risk classification tier
            if cari_score < 30.0:
                classification = "NOMINAL"
            elif 30.0 <= cari_score < 55.0:
                classification = "MODERATE STRESS"
            elif 55.0 <= cari_score < 75.0:
                classification = "HIGH RISK"
            else:
                classification = "CRITICAL RISK"

            logger.info(f"Calculated CARI score: {cari_score} -> Tier: '{classification}'")
            return cari_score, classification

        except Exception as err:
            logger.error(f"Error computing CARI score: {err}")
            raise InvalidIndexCalculationError(
                message="Failed to calculate Composite Agricultural Risk Index (CARI).",
                details={"error": str(err)},
            ) from err
