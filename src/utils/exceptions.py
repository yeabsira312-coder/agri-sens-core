"""Custom Exception Hierarchy for AGRI-SENS-CORE.

This module defines the domain-specific exception hierarchy inheriting from
the master exception class `AgriSensBaseException`.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional


class AgriSensBaseException(Exception):
    """Master base exception for all AgriSens Core errors.

    Attributes:
        message (str): Human-readable error message describing the failure.
        details (Optional[Dict[str, Any]]): Contextual metadata associated with the error.
        error_code (str): Unique identifier string for error classification.
        timestamp (datetime): UTC timestamp when the exception was instantiated.
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "AGRI_SENS_ERROR",
    ) -> None:
        """Initialize AgriSensBaseException.

        Args:
            message: Descriptive error message.
            details: Optional dictionary containing error metadata.
            error_code: Unique classification code for the error.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


    def __str__(self) -> str:
        """Return formatted error representation."""
        details_str = f" | Details: {self.details}" if self.details else ""
        return f"[{self.error_code}] {self.message}{details_str} (Occurred at: {self.timestamp.isoformat()}Z)"

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"details={self.details!r}, "
            f"error_code={self.error_code!r})"
        )


class DataIngestionError(AgriSensBaseException):
    """Raised during STAC/Satellite API connectivity, timeout, or empty payload errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "DATA_INGESTION_ERROR",
    ) -> None:
        """Initialize DataIngestionError."""
        super().__init__(message=message, details=details, error_code=error_code)


class CloudCoverageExceededError(AgriSensBaseException):
    """Raised when satellite scenes exceed specified maximum cloud coverage thresholds."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "CLOUD_COVERAGE_EXCEEDED",
    ) -> None:
        """Initialize CloudCoverageExceededError."""
        super().__init__(message=message, details=details, error_code=error_code)


class SpatialAlignmentError(AgriSensBaseException):
    """Raised when CRS reprojection, bounding box, or raster grid shape mismatches occur."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "SPATIAL_ALIGNMENT_ERROR",
    ) -> None:
        """Initialize SpatialAlignmentError."""
        super().__init__(message=message, details=details, error_code=error_code)


class InvalidIndexCalculationError(AgriSensBaseException):
    """Raised when zero-division, invalid math operations, or array dimension errors occur during index calculation."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "INVALID_INDEX_CALCULATION",
    ) -> None:
        """Initialize InvalidIndexCalculationError."""
        super().__init__(message=message, details=details, error_code=error_code)
