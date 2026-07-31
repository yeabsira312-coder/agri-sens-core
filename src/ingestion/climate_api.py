"""NASA POWER Climate API Ingestion Engine.

Fetches daily meteorological and soil moisture parameters (TS, GWETTOP) from the
NASA POWER API and formats them into time-indexed Pandas DataFrames.
"""

from typing import Any, Dict, Optional
import pandas as pd
import requests

from src.utils.exceptions import DataIngestionError
from src.utils.logger import get_logger

logger = get_logger("agri_sens.ingestion.climate")

DEFAULT_POWER_API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
DEFAULT_PARAMETERS = ["TS", "GWETTOP"]


class NASADataPowerIngestionEngine:
    """Engine for fetching point daily climate data from NASA POWER API.

    Attributes:
        api_url (str): Base URL endpoint for NASA POWER Daily Point API.
        community (str): POWER community context (default: 'AG' for Agroclimatology).
    """

    def __init__(
        self,
        api_url: str = DEFAULT_POWER_API_URL,
        community: str = "AG",
    ) -> None:
        """Initialize NASADataPowerIngestionEngine.

        Args:
            api_url: API endpoint URL.
            community: Community type for parameter context ('AG' for agroclimatology).
        """
        self.api_url = api_url
        self.community = community
        logger.info(f"Initialized NASADataPowerIngestionEngine with URL: {self.api_url}")

    def _format_date(self, date_str: str) -> str:
        """Format input date string into YYYYMMDD format required by NASA POWER API.

        Args:
            date_str: Date string in 'YYYY-MM-DD' or 'YYYYMMDD' format.

        Returns:
            Formatted date string 'YYYYMMDD'.
        """
        clean_date = date_str.replace("-", "").strip()
        if len(clean_date) != 8 or not clean_date.isdigit():
            raise DataIngestionError(
                message=f"Invalid date format: '{date_str}'. Expected 'YYYY-MM-DD' or 'YYYYMMDD'.",
                details={"date_str": date_str},
            )
        return clean_date

    def fetch_daily_climate(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        timeout: int = 30,
    ) -> pd.DataFrame:
        """Query NASA POWER Daily Point API for TS and GWETTOP parameters.

        Args:
            latitude: Latitude coordinate (-90.0 to 90.0).
            longitude: Longitude coordinate (-180.0 to 180.0).
            start_date: Start date ('YYYY-MM-DD' or 'YYYYMMDD').
            end_date: End date ('YYYY-MM-DD' or 'YYYYMMDD').
            timeout: HTTP request timeout in seconds.

        Returns:
            pandas.DataFrame containing climate parameters indexed by DatetimeIndex.
            Columns: ['TS_C', 'GWETTOP']

        Raises:
            DataIngestionError: If API request fails, times out, or returns invalid payload.
        """
        if not (-90.0 <= latitude <= 90.0) or not (-180.0 <= longitude <= 180.0):
            raise DataIngestionError(
                message=f"Invalid coordinate parameters: lat={latitude}, lon={longitude}",
                details={"latitude": latitude, "longitude": longitude},
            )

        formatted_start = self._format_date(start_date)
        formatted_end = self._format_date(end_date)

        params = {
            "parameters": ",".join(DEFAULT_PARAMETERS),
            "community": self.community,
            "longitude": longitude,
            "latitude": latitude,
            "start": formatted_start,
            "end": formatted_end,
            "format": "JSON",
        }

        logger.info(
            f"Querying NASA POWER API for lat={latitude}, lon={longitude}, "
            f"period={formatted_start}-{formatted_end}"
        )

        try:
            response = requests.get(self.api_url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as err:
            logger.error(f"NASA POWER API request timed out after {timeout} seconds.")
            raise DataIngestionError(
                message="NASA POWER API connection timed out.",
                details={"timeout": timeout, "error": str(err)},
            ) from err
        except (requests.RequestException, Exception) as err:
            logger.error(f"HTTP request error while accessing NASA POWER API: {err}")
            raise DataIngestionError(
                message=f"Failed to fetch data from NASA POWER API (HTTP error)",
                details={"api_url": self.api_url, "error": str(err)},
            ) from err

        except ValueError as err:
            logger.error("Failed to parse JSON payload from NASA POWER response.")
            raise DataIngestionError(
                message="Invalid JSON payload returned by NASA POWER API.",
                details={"error": str(err)},
            ) from err

        return self._parse_response(payload)

    def _parse_response(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """Parse raw NASA POWER API JSON payload into formatted DataFrame.

        Args:
            payload: JSON response dictionary from NASA POWER API.

        Returns:
            Parsed DataFrame with DatetimeIndex and columns ['TS_C', 'GWETTOP'].

        Raises:
            DataIngestionError: If expected parameters or parameter properties are missing.
        """
        try:
            properties = payload.get("properties", {})
            parameter_data = properties.get("parameter", {})

            if not parameter_data:
                logger.error("NASA POWER payload missing 'properties.parameter' object.")
                raise DataIngestionError(
                    message="NASA POWER API payload does not contain parameter data.",
                    details={"payload_keys": list(payload.keys())},
                )

            ts_dict = parameter_data.get("TS", {})
            gwettop_dict = parameter_data.get("GWETTOP", {})

            if not ts_dict and not gwettop_dict:
                raise DataIngestionError(
                    message="Empty parameter data returned for TS and GWETTOP in NASA POWER payload.",
                )

            # Build DataFrame from dictionaries
            df_ts = pd.DataFrame(list(ts_dict.items()), columns=["DateStr", "TS"])
            df_gwettop = pd.DataFrame(list(gwettop_dict.items()), columns=["DateStr", "GWETTOP"])

            df = pd.merge(df_ts, df_gwettop, on="DateStr", how="outer")

            if df.empty:
                raise DataIngestionError("Constructed climate DataFrame is empty.")

            # Parse dates
            df["Date"] = pd.to_datetime(df["DateStr"], format="%Y%m%d", errors="coerce")
            df = df.dropna(subset=["Date"])
            df = df.set_index("Date").sort_index()

            # Handle missing data values (-999.0 is standard NASA POWER fill value)
            df = df.replace(-999.0, float("nan"))

            # Convert Kelvin to Celsius if values are in Kelvin (>150 K)
            if df["TS"].notna().any() and df["TS"].mean() > 150.0:
                logger.info("Converting Land Surface Temperature (TS) from Kelvin to Celsius.")
                df["TS_C"] = df["TS"] - 273.15
            else:
                df["TS_C"] = df["TS"]

            df = df.drop(columns=["DateStr", "TS"], errors="ignore")
            df = df[["TS_C", "GWETTOP"]]

            logger.info(f"Successfully parsed NASA POWER data ({len(df)} daily records).")
            return df

        except DataIngestionError:
            raise
        except Exception as err:
            logger.error(f"Unexpected error parsing NASA POWER response: {err}")
            raise DataIngestionError(
                message="Error parsing NASA POWER response structure.",
                details={"error": str(err)},
            ) from err
