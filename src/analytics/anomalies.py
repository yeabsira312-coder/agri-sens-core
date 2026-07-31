"""Climate Anomaly Detection Engine Module.

Computes calendar-week rolling Z-scores for Soil Moisture (Z_SM) and Land Surface
Temperature (Z_LST), flagging Drought Risk and Heat Stress Risk anomalies.
"""

from typing import Optional
import numpy as np
import pandas as pd

from src.utils.exceptions import InvalidIndexCalculationError
from src.utils.logger import get_logger

logger = get_logger("agri_sens.analytics.anomalies")


class AnomalyDetectionEngine:
    """Engine for computing multi-year baseline climate Z-scores and stress flags."""

    @classmethod
    def calculate_weekly_z_scores(
        cls,
        df: pd.DataFrame,
        rolling_window_years: int = 10,
        sm_col: str = "GWETTOP",
        lst_col: str = "TS_C",
    ) -> pd.DataFrame:
        """Calculate weekly Z-scores for Soil Moisture and Land Surface Temperature.

        Args:
            df: Input pandas.DataFrame indexed by DatetimeIndex, containing sm_col and lst_col.
            rolling_window_years: Rolling baseline window size in years (default: 10).
            sm_col: Soil moisture column name (default: 'GWETTOP').
            lst_col: Land surface temperature column name (default: 'TS_C').

        Returns:
            pandas.DataFrame augmented with columns:
            ['Z_SM', 'Z_LST', 'Drought_Risk', 'Heat_Stress_Risk']

        Raises:
            InvalidIndexCalculationError: If DataFrame lacks DatetimeIndex or required columns.
        """
        logger.info(f"Computing weekly Z-scores for climate variables (sm_col='{sm_col}', lst_col='{lst_col}')")

        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("Input DataFrame index is not a DatetimeIndex.")
            raise InvalidIndexCalculationError(
                message="Input DataFrame must be indexed by a pandas DatetimeIndex.",
                details={"index_type": str(type(df.index))},
            )

        missing_cols = [c for c in [sm_col, lst_col] if c not in df.columns]
        if missing_cols:
            logger.error(f"Missing required climate columns: {missing_cols}")
            raise InvalidIndexCalculationError(
                message=f"DataFrame is missing required climate columns: {missing_cols}",
                details={"df_columns": list(df.columns)},
            )

        if df.empty:
            raise InvalidIndexCalculationError(message="Cannot compute Z-scores on empty DataFrame.")

        out_df = df.copy()

        try:
            # Extract calendar week number (1 to 53)
            out_df["Week"] = out_df.index.isocalendar().week.astype(int)

            # Compute weekly baseline mean and std dev
            weekly_stats = out_df.groupby("Week")[[sm_col, lst_col]].agg(["mean", "std"])

            # Map stats back to DataFrame rows based on Week
            out_df["SM_mean"] = out_df["Week"].map(weekly_stats[(sm_col, "mean")])
            out_df["SM_std"] = out_df["Week"].map(weekly_stats[(sm_col, "std")])
            out_df["LST_mean"] = out_df["Week"].map(weekly_stats[(lst_col, "mean")])
            out_df["LST_std"] = out_df["Week"].map(weekly_stats[(lst_col, "std")])

            # Calculate Z-scores with zero-std protection
            with np.errstate(divide="ignore", invalid="ignore"):
                # Z_SM = (GWETTOP - mean) / std
                out_df["Z_SM"] = np.where(
                    (out_df["SM_std"].isna()) | (out_df["SM_std"] == 0.0),
                    0.0,
                    (out_df[sm_col] - out_df["SM_mean"]) / out_df["SM_std"],
                )

                # Z_LST = (TS_C - mean) / std
                out_df["Z_LST"] = np.where(
                    (out_df["LST_std"].isna()) | (out_df["LST_std"] == 0.0),
                    0.0,
                    (out_df[lst_col] - out_df["LST_mean"]) / out_df["LST_std"],
                )

            # Flag risks
            # Drought Risk: Z_SM < -1.5 (Soil moisture >1.5 std below average)
            # Heat Stress Risk: Z_LST > +1.5 (Temperature >1.5 std above average)
            out_df["Drought_Risk"] = out_df["Z_SM"] < -1.5
            out_df["Heat_Stress_Risk"] = out_df["Z_LST"] > 1.5

            # Clean up internal temporary columns
            drop_cols = ["Week", "SM_mean", "SM_std", "LST_mean", "LST_std"]
            out_df = out_df.drop(columns=drop_cols, errors="ignore")

            logger.info("Successfully computed weekly climate Z-scores and risk flags.")
            return out_df

        except Exception as err:
            logger.error(f"Failed to calculate weekly Z-scores: {err}")
            raise InvalidIndexCalculationError(
                message="Error calculating weekly climate Z-scores.",
                details={"error": str(err)},
            ) from err
