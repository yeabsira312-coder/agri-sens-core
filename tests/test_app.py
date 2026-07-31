"""Unit and integration tests for AGRI-SENS-CORE Step 5 Web App Components."""

import unittest
import pandas as pd

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    folium = None
    HAS_FOLIUM = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    go = None
    HAS_PLOTLY = False

from app.components.maps import create_risk_map
from app.components.plots import create_climate_anomaly_plot, create_ndvi_timeseries_plot
from app.streamlit_app import run_cached_pipeline
from app.views.tab_export import generate_policy_brief


class TestAppComponents(unittest.TestCase):
    """Test suite for Streamlit web app visualization components and data generators."""

    def setUp(self) -> None:
        """Set up test environment and synthetic dataframe."""
        dates = pd.date_range("2023-01-01", "2023-01-30", freq="D")
        self.df = pd.DataFrame(
            {
                "NDVI": [0.6] * len(dates),
                "NDVI_Baseline": [0.6] * len(dates),
                "NDVI_Std": [0.05] * len(dates),
                "Z_SM": [0.0] * len(dates),
                "Z_LST": [0.0] * len(dates),
            },
            index=dates,
        )
        self.center_lat = -1.28
        self.center_lon = 36.82
        self.bbox = (36.77, -1.33, 36.87, -1.23)

    @unittest.skipUnless(HAS_FOLIUM, "folium package required for map test")
    def test_create_risk_map_generation(self) -> None:
        """Verify create_risk_map returns a valid Folium Map object."""
        m = create_risk_map(
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            bbox=self.bbox,
            cari_score=42.5,
            risk_tier="MODERATE STRESS",
        )
        self.assertIsInstance(m, folium.Map)
        self.assertEqual(m.location, [self.center_lat, self.center_lon])

    @unittest.skipUnless(HAS_PLOTLY, "plotly package required for plot test")
    def test_create_ndvi_timeseries_plot_generation(self) -> None:
        """Verify create_ndvi_timeseries_plot returns a valid Plotly Figure."""
        fig = create_ndvi_timeseries_plot(self.df)
        self.assertIsInstance(fig, go.Figure)
        self.assertGreaterEqual(len(fig.data), 1)

    @unittest.skipUnless(HAS_PLOTLY, "plotly package required for plot test")
    def test_create_climate_anomaly_plot_generation(self) -> None:
        """Verify create_climate_anomaly_plot returns a valid Plotly Figure."""
        fig = create_climate_anomaly_plot(self.df)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)


    def test_generate_policy_brief_text(self) -> None:
        """Verify generate_policy_brief outputs structured policy brief text."""
        brief = generate_policy_brief(
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            cari_score=78.2,
            risk_tier="CRITICAL RISK",
            ndvi_obs=0.25,
            z_sm=-2.5,
            z_lst=2.8,
        )

        self.assertIsInstance(brief, str)
        self.assertIn("CRITICAL RISK", brief)
        self.assertIn("CRITICAL EMERGENCY ALERT", brief)

    def test_run_cached_pipeline_execution(self) -> None:
        """Verify run_cached_pipeline generates valid DataFrame with required analytics columns."""
        df = run_cached_pipeline(
            lat=self.center_lat,
            lon=self.center_lon,
            start_date_str="2023-01-01",
            end_date_str="2023-01-20",
            max_cloud=10.0,
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("Z_SM", df.columns)
        self.assertIn("Z_LST", df.columns)
        self.assertEqual(len(df), 20)


if __name__ == "__main__":
    unittest.main()
