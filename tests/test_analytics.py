"""Unit tests for AGRI-SENS-CORE Step 4 Scientific Feature Engineering & Risk Engine."""

import math
import unittest
import numpy as np
import pandas as pd

from src.analytics.anomalies import AnomalyDetectionEngine
from src.analytics.indices import VegetationIndexEngine
from src.analytics.risk_engine import RiskEngine
from src.utils.exceptions import InvalidIndexCalculationError


class TestVegetationIndexEngine(unittest.TestCase):
    """Test suite for VegetationIndexEngine."""

    def test_compute_ndvi_scalar(self) -> None:
        """Verify NDVI scalar calculation and formula."""
        # NIR=0.8, RED=0.2 -> (0.8 - 0.2) / (0.8 + 0.2) = 0.6 / 1.0 = 0.6
        ndvi = VegetationIndexEngine.compute_ndvi(nir=0.8, red=0.2)
        self.assertAlmostEqual(ndvi, 0.6, places=4)

    def test_compute_ndvi_zero_division(self) -> None:
        """Verify zero denominator produces NaN without crashing."""
        ndvi = VegetationIndexEngine.compute_ndvi(nir=0.0, red=0.0)
        self.assertTrue(math.isnan(ndvi))

    def test_compute_ndvi_array_bounds(self) -> None:
        """Verify array computation and clipping bounds [-1.0, 1.0]."""
        nir = np.array([0.5, 0.9, 0.1])
        red = np.array([0.1, 0.1, 0.9])
        ndvi = VegetationIndexEngine.compute_ndvi(nir, red)

        self.assertEqual(len(ndvi), 3)
        self.assertTrue(np.all(ndvi >= -1.0))
        self.assertTrue(np.all(ndvi <= 1.0))
        self.assertAlmostEqual(ndvi[0], 0.6667, places=3)

    def test_compute_evi_scalar(self) -> None:
        """Verify EVI scalar calculation."""
        # NIR=0.5, RED=0.1, BLUE=0.05
        evi = VegetationIndexEngine.compute_evi(nir=0.5, red=0.1, blue=0.05)
        self.assertIsInstance(evi, float)
        self.assertGreater(evi, 0.0)

    def test_compute_savi_scalar(self) -> None:
        """Verify SAVI scalar calculation with default L=0.5."""
        # NIR=0.6, RED=0.2, L=0.5 -> ((0.6-0.2)/(0.6+0.2+0.5)) * 1.5 = (0.4 / 1.3) * 1.5 = 0.4615
        savi = VegetationIndexEngine.compute_savi(nir=0.6, red=0.2, l_factor=0.5)
        self.assertAlmostEqual(savi, 0.4615, places=3)

    def test_shape_mismatch_raises_error(self) -> None:
        """Verify input array shape mismatch raises InvalidIndexCalculationError."""
        nir = np.array([0.5, 0.6])
        red = np.array([0.1, 0.2, 0.3])
        with self.assertRaises(InvalidIndexCalculationError):
            VegetationIndexEngine.compute_ndvi(nir, red)


class TestAnomalyDetectionEngine(unittest.TestCase):
    """Test suite for AnomalyDetectionEngine."""

    def setUp(self) -> None:
        """Set up synthetic climate DataFrame spanning 3 years."""
        dates = pd.date_range("2021-01-01", "2023-12-31", freq="D")
        np.random.seed(42)

        # Baseline climate data
        gwettop = np.random.normal(loc=0.4, scale=0.1, size=len(dates))
        ts_c = np.random.normal(loc=25.0, scale=3.0, size=len(dates))

        self.df = pd.DataFrame(
            {"GWETTOP": gwettop, "TS_C": ts_c},
            index=dates,
        )

    def test_calculate_weekly_z_scores_structure(self) -> None:
        """Verify output columns and Z-score calculation."""
        out_df = AnomalyDetectionEngine.calculate_weekly_z_scores(self.df)

        self.assertIn("Z_SM", out_df.columns)
        self.assertIn("Z_LST", out_df.columns)
        self.assertIn("Drought_Risk", out_df.columns)
        self.assertIn("Heat_Stress_Risk", out_df.columns)

        # Mean of Z-scores should be approximately 0.0
        self.assertAlmostEqual(out_df["Z_SM"].mean(), 0.0, places=1)
        self.assertAlmostEqual(out_df["Z_LST"].mean(), 0.0, places=1)

    def test_risk_flags_detection(self) -> None:
        """Verify drought and heat stress risk flags trigger correctly."""
        # Inject severe dry & hot anomaly on specific date
        anomaly_date = "2023-06-15"
        self.df.loc[anomaly_date, "GWETTOP"] = 0.01  # Extremely low soil moisture
        self.df.loc[anomaly_date, "TS_C"] = 45.0  # Extremely high temperature

        out_df = AnomalyDetectionEngine.calculate_weekly_z_scores(self.df)

        self.assertTrue(out_df.loc[anomaly_date, "Drought_Risk"])
        self.assertTrue(out_df.loc[anomaly_date, "Heat_Stress_Risk"])
        self.assertLess(out_df.loc[anomaly_date, "Z_SM"], -1.5)
        self.assertGreater(out_df.loc[anomaly_date, "Z_LST"], 1.5)

    def test_missing_index_raises_error(self) -> None:
        """Verify missing DatetimeIndex raises InvalidIndexCalculationError."""
        df_no_dt = self.df.reset_index(drop=True)
        with self.assertRaises(InvalidIndexCalculationError):
            AnomalyDetectionEngine.calculate_weekly_z_scores(df_no_dt)


class TestRiskEngine(unittest.TestCase):
    """Test suite for RiskEngine (CARI calculation)."""

    def test_logistic_transform_bounds(self) -> None:
        """Verify logistic transform returns values strictly bounded in (0, 1)."""
        self.assertAlmostEqual(RiskEngine.logistic_transform(0.0), 0.5, places=4)
        self.assertGreater(RiskEngine.logistic_transform(10.0), 0.99)
        self.assertLess(RiskEngine.logistic_transform(-10.0), 0.01)

    def test_compute_cari_nominal_conditions(self) -> None:
        """Verify nominal healthy field condition returns low CARI and NOMINAL tier."""
        # Healthy canopy (+20% canopy growth), high soil moisture (+2 std), cool surface temp (-2 std)
        score, tier = RiskEngine.compute_cari(
            ndvi_obs=0.72,
            ndvi_baseline=0.60,
            z_sm=2.0,
            z_lst=-2.0,
        )

        self.assertIsInstance(score, float)
        self.assertLess(score, 30.0)
        self.assertEqual(tier, "NOMINAL")

    def test_compute_cari_critical_stress_conditions(self) -> None:
        """Verify severe drought & canopy loss condition returns CRITICAL RISK tier."""
        # Severe canopy drop (obs=0.2, baseline=0.6 -> -66% drop), severe dry (Z_SM=-3.0), extreme heat (Z_LST=+3.0)
        score, tier = RiskEngine.compute_cari(
            ndvi_obs=0.2,
            ndvi_baseline=0.6,
            z_sm=-3.0,
            z_lst=3.0,
        )

        self.assertGreaterEqual(score, 75.0)
        self.assertEqual(tier, "CRITICAL RISK")

    def test_compute_cari_classification_tiers(self) -> None:
        """Verify all risk classification boundaries."""
        # Nominal case
        _, tier_nominal = RiskEngine.compute_cari(0.72, 0.60, 2.0, -2.0)
        self.assertEqual(tier_nominal, "NOMINAL")

        # Moderate Stress case
        _, tier_moderate = RiskEngine.compute_cari(0.60, 0.60, 0.0, 0.0)
        self.assertEqual(tier_moderate, "MODERATE STRESS")


    def test_invalid_nan_inputs_raises_error(self) -> None:
        """Verify NaN inputs raise InvalidIndexCalculationError."""
        with self.assertRaises(InvalidIndexCalculationError):
            RiskEngine.compute_cari(float("nan"), 0.7, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
