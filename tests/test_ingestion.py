"""Unit tests for AGRI-SENS-CORE Step 2 Data Ingestion Pipeline."""

import math
from typing import Any, Dict
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    xr = MagicMock()
    HAS_XARRAY = False

from src.ingestion.climate_api import NASADataPowerIngestionEngine
from src.ingestion.stac_client import SentinelSTACIngestionEngine
from src.utils.exceptions import CloudCoverageExceededError, DataIngestionError



class TestSentinelSTACIngestionEngine(unittest.TestCase):
    """Test suite for SentinelSTACIngestionEngine."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.engine = SentinelSTACIngestionEngine()
        self.sample_bbox = (36.8, -1.3, 37.0, -1.1)
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-31"

    def test_initialization(self) -> None:
        """Verify engine initialization defaults."""
        self.assertIn("planetarycomputer", self.engine.stac_url)
        self.assertEqual(self.engine.collection, "sentinel-2-l2a")

    @patch("src.ingestion.stac_client.STACClient")
    @patch("src.ingestion.stac_client.HAS_STAC", True)
    def test_search_scenes_empty_raises_error(self, mock_stac_client_cls: MagicMock) -> None:
        """Verify DataIngestionError is raised when STAC search returns no items."""
        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_search.item_collection.return_value = []
        mock_client.search.return_value = mock_search
        mock_stac_client_cls.open.return_value = mock_client

        with self.assertRaises(DataIngestionError) as ctx:
            self.engine.search_scenes(self.sample_bbox, self.start_date, self.end_date)
        self.assertIn("No STAC scenes found", ctx.exception.message)

    @patch("src.ingestion.stac_client.STACClient")
    @patch("src.ingestion.stac_client.HAS_STAC", True)
    def test_search_scenes_cloud_cover_exceeded(self, mock_stac_client_cls: MagicMock) -> None:
        """Verify CloudCoverageExceededError is raised when all scenes exceed cloud limit."""
        mock_client = MagicMock()

        # First query (with filter) returns 0 items
        mock_filtered_search = MagicMock()
        mock_filtered_search.item_collection.return_value = []

        # Second query (unfiltered) returns item with 50% cloud cover
        mock_cloudy_item = MagicMock()
        mock_cloudy_item.properties = {"eo:cloud_cover": 50.0}
        mock_unfiltered_search = MagicMock()
        mock_unfiltered_search.item_collection.return_value = [mock_cloudy_item]

        mock_client.search.side_effect = [mock_filtered_search, mock_unfiltered_search]
        mock_stac_client_cls.open.return_value = mock_client

        with self.assertRaises(CloudCoverageExceededError) as ctx:
            self.engine.search_scenes(self.sample_bbox, self.start_date, self.end_date, max_cloud_cover=10.0)
        self.assertIn("cloud cover threshold", ctx.exception.message)

    @patch("src.ingestion.stac_client.STACClient")
    @patch("src.ingestion.stac_client.HAS_STAC", True)
    def test_search_scenes_success(self, mock_stac_client_cls: MagicMock) -> None:
        """Verify successful STAC scene search returning item collection."""
        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_item = MagicMock()
        mock_search.item_collection.return_value = [mock_item]
        mock_client.search.return_value = mock_search
        mock_stac_client_cls.open.return_value = mock_client

        items = self.engine.search_scenes(self.sample_bbox, self.start_date, self.end_date)
        self.assertEqual(len(items), 1)

    def test_load_dataset_empty_items_raises_error(self) -> None:
        """Verify loading dataset from empty items raises DataIngestionError."""
        with self.assertRaises(DataIngestionError):
            self.engine.load_dataset([])

    @unittest.skipUnless(HAS_XARRAY, "xarray package required for dataset creation test")
    def test_load_dataset_success(self) -> None:
        """Verify loading xarray Dataset from mock STAC items."""
        mock_item = MagicMock()
        mock_item.id = "S2A_TEST_ITEM"
        mock_item.properties = {"datetime": "2023-01-15T10:00:00Z"}
        mock_asset = MagicMock()
        mock_asset.href = "https://example.com/B02.tif"
        mock_item.assets = {"B02": mock_asset, "B03": mock_asset, "B04": mock_asset, "B08": mock_asset, "SCL": mock_asset}

        ds = self.engine.load_dataset([mock_item], bands=("B02", "B03", "B04", "B08", "SCL"))
        self.assertIsInstance(ds, xr.Dataset)
        self.assertIn("B02", ds.data_vars)
        self.assertIn("B08", ds.data_vars)
        self.assertEqual(ds.attrs["items_count"], 1)



class TestNASADataPowerIngestionEngine(unittest.TestCase):
    """Test suite for NASADataPowerIngestionEngine."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.engine = NASADataPowerIngestionEngine()
        self.latitude = -1.28
        self.longitude = 36.82
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-05"

    def test_date_formatting(self) -> None:
        """Verify date format conversion to YYYYMMDD."""
        self.assertEqual(self.engine._format_date("2023-01-15"), "20230115")
        self.assertEqual(self.engine._format_date("20230115"), "20230115")

        with self.assertRaises(DataIngestionError):
            self.engine._format_date("invalid-date")

    def test_invalid_coordinates_raises_error(self) -> None:
        """Verify invalid lat/lon coordinates raise DataIngestionError."""
        with self.assertRaises(DataIngestionError):
            self.engine.fetch_daily_climate(latitude=100.0, longitude=36.0, start_date=self.start_date, end_date=self.end_date)

    @patch("src.ingestion.climate_api.requests.get")
    def test_fetch_daily_climate_success_and_kelvin_conversion(self, mock_get: MagicMock) -> None:
        """Verify successful climate API fetching and Kelvin to Celsius conversion."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "properties": {
                "parameter": {
                    "TS": {"20230101": 300.15, "20230102": 302.15},  # Kelvin values
                    "GWETTOP": {"20230101": 0.45, "20230102": 0.48},
                }
            }
        }
        mock_get.return_value = mock_response

        df = self.engine.fetch_daily_climate(
            latitude=self.latitude,
            longitude=self.longitude,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertIn("TS_C", df.columns)
        self.assertIn("GWETTOP", df.columns)
        self.assertEqual(len(df), 2)

        # Check Kelvin to Celsius conversion (300.15 K - 273.15 = 27.0 °C)
        self.assertAlmostEqual(df.loc["2023-01-01", "TS_C"], 27.0, places=2)
        self.assertAlmostEqual(df.loc["2023-01-02", "TS_C"], 29.0, places=2)

    @patch("src.ingestion.climate_api.requests.get")
    def test_fetch_daily_climate_http_error_raises_exception(self, mock_get: MagicMock) -> None:
        """Verify HTTP error raises DataIngestionError."""
        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_get.return_value = mock_response

        with self.assertRaises(DataIngestionError) as ctx:
            self.engine.fetch_daily_climate(
                latitude=self.latitude,
                longitude=self.longitude,
                start_date=self.start_date,
                end_date=self.end_date,
            )
        self.assertIn("HTTP error", ctx.exception.message)



if __name__ == "__main__":
    unittest.main()
