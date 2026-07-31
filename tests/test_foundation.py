"""Unit tests for AGRI-SENS-CORE Step 1 Foundation, Logging & Custom Exceptions."""

import logging
from pathlib import Path
import tempfile
import unittest

from src.utils.exceptions import (
    AgriSensBaseException,
    CloudCoverageExceededError,
    DataIngestionError,
    InvalidIndexCalculationError,
    SpatialAlignmentError,
)
from src.utils.logger import (
    critical,
    debug,
    error,
    get_logger,
    info,
    setup_logger,
    warning,
)


class TestCustomExceptions(unittest.TestCase):
    """Test suite for custom exception classes."""

    def test_base_exception_attributes(self) -> None:
        """Verify AgriSensBaseException attributes and string formatting."""
        exc = AgriSensBaseException(
            message="Test error message",
            details={"key": "value"},
            error_code="TEST_CODE",
        )
        self.assertEqual(exc.message, "Test error message")
        self.assertEqual(exc.details, {"key": "value"})
        self.assertEqual(exc.error_code, "TEST_CODE")
        self.assertIsNotNone(exc.timestamp)
        self.assertIn("[TEST_CODE] Test error message", str(exc))
        self.assertIn("Details: {'key': 'value'}", str(exc))

    def test_data_ingestion_error_inheritance(self) -> None:
        """Verify DataIngestionError inherits from AgriSensBaseException."""
        exc = DataIngestionError("STAC API connection timed out", details={"url": "http://stac.api"})
        self.assertIsInstance(exc, AgriSensBaseException)
        self.assertEqual(exc.error_code, "DATA_INGESTION_ERROR")
        self.assertEqual(exc.details["url"], "http://stac.api")

    def test_cloud_coverage_exceeded_error(self) -> None:
        """Verify CloudCoverageExceededError attributes."""
        exc = CloudCoverageExceededError("Cloud cover 45% exceeds threshold 15%", details={"cloud_cover": 45})
        self.assertIsInstance(exc, AgriSensBaseException)
        self.assertEqual(exc.error_code, "CLOUD_COVERAGE_EXCEEDED")

    def test_spatial_alignment_error(self) -> None:
        """Verify SpatialAlignmentError attributes."""
        exc = SpatialAlignmentError("CRS mismatch between EPSG:4326 and EPSG:32636")
        self.assertIsInstance(exc, AgriSensBaseException)
        self.assertEqual(exc.error_code, "SPATIAL_ALIGNMENT_ERROR")

    def test_invalid_index_calculation_error(self) -> None:
        """Verify InvalidIndexCalculationError attributes."""
        exc = InvalidIndexCalculationError("Division by zero encountered in NDVI calculation")
        self.assertIsInstance(exc, AgriSensBaseException)
        self.assertEqual(exc.error_code, "INVALID_INDEX_CALCULATION")

    def test_exception_catching_polymorphism(self) -> None:
        """Verify all specific exceptions are caught by AgriSensBaseException handler."""
        exceptions_to_test = [
            DataIngestionError("API Failure"),
            CloudCoverageExceededError("Too cloudy"),
            SpatialAlignmentError("CRS mismatch"),
            InvalidIndexCalculationError("Zero division"),
        ]

        for exc in exceptions_to_test:
            with self.assertRaises(AgriSensBaseException) as ctx:
                raise exc
            self.assertIsNotNone(ctx.exception.message)


class TestLoggingSystem(unittest.TestCase):
    """Test suite for logging module."""

    def test_logger_initialization(self) -> None:
        """Verify setup_logger initializes console and file handlers correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir) / "test_logs"
            logger = setup_logger(name="test_logger", log_dir=str(log_dir), log_level=logging.DEBUG)

            self.assertEqual(logger.name, "test_logger")
            self.assertGreaterEqual(len(logger.handlers), 2)

            # Log a test message
            logger.info("Test info message")
            log_file = log_dir / "agri_sens.log"
            self.assertTrue(log_file.exists())

            content = log_file.read_text(encoding="utf-8")
            self.assertIn("Test info message", content)

            # Close handlers so Windows can release file lock
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


    def test_module_logging_helpers(self) -> None:
        """Verify module level helper functions execute without errors."""
        info("Module info test message")
        warning("Module warning test message")
        error("Module error test message")
        debug("Module debug test message")
        critical("Module critical test message")

        # Verify log file exists and contains messages
        project_root = Path(__file__).resolve().parent.parent
        log_file = project_root / "logs" / "agri_sens.log"
        self.assertTrue(log_file.exists())


if __name__ == "__main__":
    unittest.main()

