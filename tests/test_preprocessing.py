"""Unit tests for AGRI-SENS-CORE Step 3 Geospatial Preprocessing & Raster Operations."""

from typing import Any
import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    xr = MagicMock()
    HAS_XARRAY = False

from src.preprocessing.cloud_masking import mask_sentinel2_clouds
from src.preprocessing.reproject import align_spatial_grid
from src.preprocessing.temporal import create_10day_median_composites
from src.utils.exceptions import SpatialAlignmentError


class TestCloudMasking(unittest.TestCase):
    """Test suite for SCL cloud masking functions."""

    @unittest.skipUnless(HAS_XARRAY, "xarray package required for cloud masking test")
    def test_mask_sentinel2_clouds_filtering(self) -> None:
        """Verify invalid SCL values (clouds, shadows) are masked to NaN."""
        # Create synthetic 2x2 raster across 1 time step
        # SCL pixel values: [[4 (Veg), 8 (Cloud Medium)], [9 (Cloud High), 5 (Soil)]]
        scl_data = np.array([[[4, 8], [9, 5]]], dtype=np.int32)
        b04_data = np.array([[[1000.0, 2000.0], [3000.0, 4000.0]]], dtype=np.float32)

        ds = xr.Dataset(
            data_vars={
                "SCL": (("time", "y", "x"), scl_data),
                "B04": (("time", "y", "x"), b04_data),
            },
            coords={"time": pd.date_range("2023-01-01", periods=1)},
        )

        masked_ds = mask_sentinel2_clouds(ds)

        # Check that Vegetation (4) and Soil (5) retain original values
        self.assertEqual(masked_ds["B04"].values[0, 0, 0], 1000.0)
        self.assertEqual(masked_ds["B04"].values[0, 1, 1], 4000.0)

        # Check that Cloud Medium (8) and Cloud High (9) become NaN
        self.assertTrue(np.isnan(masked_ds["B04"].values[0, 0, 1]))
        self.assertTrue(np.isnan(masked_ds["B04"].values[0, 1, 0]))

    def test_mask_sentinel2_clouds_missing_scl_raises_error(self) -> None:
        """Verify missing SCL band raises SpatialAlignmentError."""
        if not HAS_XARRAY:
            self.skipTest("xarray package required")

        ds = xr.Dataset(
            data_vars={"B04": (("y", "x"), np.ones((2, 2)))},
        )

        with self.assertRaises(SpatialAlignmentError) as ctx:
            mask_sentinel2_clouds(ds)
        self.assertIn("missing required Scene Classification Layer", ctx.exception.message)


class TestSpatialReprojection(unittest.TestCase):
    """Test suite for spatial grid reprojection and alignment."""

    @unittest.skipUnless(HAS_XARRAY, "xarray package required for reprojection test")
    def test_align_spatial_grid_metadata(self) -> None:
        """Verify grid alignment function assigns target CRS metadata."""
        ds = xr.Dataset(
            data_vars={"B04": (("y", "x"), np.ones((2, 2)))},
        )

        reprojected_ds = align_spatial_grid(ds, target_crs="EPSG:4326", resolution=0.001)
        self.assertIsNotNone(reprojected_ds)
        self.assertEqual(reprojected_ds.attrs.get("crs"), "EPSG:4326")

    def test_align_spatial_grid_invalid_object_raises_error(self) -> None:
        """Verify passing non-dataset object raises SpatialAlignmentError."""
        with self.assertRaises(SpatialAlignmentError):
            align_spatial_grid("not_a_dataset")


class TestTemporalCompositing(unittest.TestCase):
    """Test suite for 10-day median temporal compositing."""

    @unittest.skipUnless(HAS_XARRAY, "xarray package required for temporal compositing test")
    def test_create_10day_median_composites_aggregation(self) -> None:
        """Verify 10-day median compositing correctly aggregates irregular time series."""
        # 6 daily observations over 25 days
        dates = pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-08", "2023-01-12", "2023-01-15", "2023-01-22"])

        # Values for pixel (0,0): [10, 20, 30] in window 1, [100, NaN] in window 2, [500] in window 3
        data = np.array([10.0, 20.0, 30.0, 100.0, np.nan, 500.0]).reshape((6, 1, 1))

        ds = xr.Dataset(
            data_vars={"B08": (("time", "y", "x"), data)},
            coords={"time": dates},
        )

        composites = create_10day_median_composites(ds, freq="10D")

        self.assertIsInstance(composites, xr.Dataset)
        # Should create 3 composite periods (Jan 01-10, Jan 11-20, Jan 21-30)
        self.assertEqual(len(composites["time"]), 3)

        # Window 1 median of [10, 20, 30] = 20.0
        self.assertEqual(composites["B08"].values[0, 0, 0], 20.0)

        # Window 2 median of [100, NaN] with skipna=True = 100.0
        self.assertEqual(composites["B08"].values[1, 0, 0], 100.0)

    def test_create_10day_median_composites_missing_time_raises_error(self) -> None:
        """Verify dataset missing 'time' dimension raises SpatialAlignmentError."""
        if not HAS_XARRAY:
            self.skipTest("xarray package required")

        ds = xr.Dataset(
            data_vars={"B08": (("y", "x"), np.ones((2, 2)))},
        )

        with self.assertRaises(SpatialAlignmentError) as ctx:
            create_10day_median_composites(ds)
        self.assertIn("missing required temporal dimension", ctx.exception.message)


if __name__ == "__main__":
    unittest.main()
