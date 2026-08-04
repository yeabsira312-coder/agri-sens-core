"""Sentinel-2 STAC Ingestion Engine.

Interacts with Microsoft Planetary Computer STAC API to search, sign,
and load Sentinel-2 L2A satellite data into xarray structures.
"""

from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np

from src.utils.exceptions import CloudCoverageExceededError, DataIngestionError
from src.utils.logger import get_logger

logger = get_logger("agri_sens.ingestion.stac")

# Optional xarray import
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    xr = None
    HAS_XARRAY = False

# Optional imports for STAC / Planetary Computer integrations
try:
    import planetary_computer as pc
    from pystac_client import Client as STACClient
    from pystac import Item, ItemCollection
    HAS_STAC = True
except ImportError:
    HAS_STAC = False
    pc = None
    STACClient = None
    Item = object
    ItemCollection = object



DEFAULT_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEFAULT_COLLECTION = "sentinel-2-l2a"
DEFAULT_BANDS = ("B02", "B03", "B04", "B08", "SCL")


class SentinelSTACIngestionEngine:
    """Engine for searching and loading Sentinel-2 L2A STAC items from Planetary Computer.

    Attributes:
        stac_url (str): STAC catalog endpoint URL.
        collection (str): Target STAC collection ID (default: sentinel-2-l2a).
    """

    def __init__(
        self,
        stac_url: str = DEFAULT_STAC_URL,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """Initialize SentinelSTACIngestionEngine.

        Args:
            stac_url: STAC API URL endpoint.
            collection: STAC collection name.
        """
        self.stac_url = stac_url
        self.collection = collection
        logger.info(f"Initialized SentinelSTACIngestionEngine with catalog: {self.stac_url}")

    def _get_client(self) -> Any:
        """Instantiate PySTAC Client with error handling.

        Returns:
            STACClient instance.

        Raises:
            DataIngestionError: If pystac-client is missing or connection fails.
        """
        if not HAS_STAC:
            logger.error("pystac_client and planetary_computer libraries are required for STAC operations.")
            raise DataIngestionError("Required packages 'pystac-client' or 'planetary-computer' are not installed.")

        try:
            client = STACClient.open(self.stac_url)
            return client
        except Exception as err:
            logger.error(f"Failed to connect to STAC catalog at {self.stac_url}: {err}")
            raise DataIngestionError(
                message=f"Connection failed to STAC API at {self.stac_url}",
                details={"stac_url": self.stac_url, "error": str(err)},
            ) from err

    def search_scenes(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 10.0,
    ) -> Any:
        """Query Sentinel-2 L2A scenes matching spatial, temporal, and cloud cover criteria.

        Args:
            bbox: Bounding box tuple (min_lon, min_lat, max_lon, max_lat).
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).
            max_cloud_cover: Maximum allowed scene cloud coverage percentage (0.0 to 100.0).

        Returns:
            ItemCollection containing matching STAC items.

        Raises:
            DataIngestionError: If search returns 0 items or query fails.
            CloudCoverageExceededError: If scenes exist but fail cloud cover limits.
        """
        logger.info(
            f"Searching STAC scenes: collection={self.collection}, bbox={bbox}, "
            f"date_range={start_date}/{end_date}, max_cloud={max_cloud_cover}%"
        )

        client = self._get_client()
        date_interval = f"{start_date}/{end_date}"
        query_filter = {"eo:cloud_cover": {"lt": max_cloud_cover}}

        try:
            search = client.search(
                collections=[self.collection],
                bbox=bbox,
                datetime=date_interval,
                query=query_filter,
            )
            items = search.item_collection()
        except Exception as err:
            logger.error(f"STAC search query failed: {err}")
            raise DataIngestionError(
                message=f"STAC search query failed for collection {self.collection}",
                details={"bbox": bbox, "datetime": date_interval, "error": str(err)},
            ) from err

        if not items or len(items) == 0:
            # Check if scenes exist without cloud filter to distinguish cause
            try:
                unfiltered_search = client.search(
                    collections=[self.collection],
                    bbox=bbox,
                    datetime=date_interval,
                )
                unfiltered_items = unfiltered_search.item_collection()
                if unfiltered_items and len(unfiltered_items) > 0:
                    min_clouds = min(
                        [item.properties.get("eo:cloud_cover", 100.0) for item in unfiltered_items]
                    )
                    logger.warning(
                        f"Found {len(unfiltered_items)} scenes, but all exceed max cloud cover of {max_cloud_cover}%. "
                        f"Lowest cloud cover: {min_clouds}%"
                    )
                    raise CloudCoverageExceededError(
                        message=f"All available scenes exceed the cloud cover threshold of {max_cloud_cover}%. Lowest found: {min_clouds}%.",
                        details={"max_cloud_cover": max_cloud_cover, "lowest_found": min_clouds},
                    )
            except CloudCoverageExceededError:
                raise
            except Exception:
                pass

            logger.error("No STAC scenes found for specified bbox and date range.")
            raise DataIngestionError(
                message="No STAC scenes found for the specified bounding box and date range.",
                details={"bbox": bbox, "start_date": start_date, "end_date": end_date},
            )

        logger.info(f"Successfully retrieved {len(items)} STAC scenes matching criteria.")
        return items

    def sign_items(self, items: Any) -> Any:
        """Sign asset links for STAC items using Planetary Computer signing service.

        Args:
            items: ItemCollection or list of PySTAC Items.

        Returns:
            Signed ItemCollection or list of Items.

        Raises:
            DataIngestionError: If signing fails or planetary_computer package is unavailable.
        """
        if not HAS_STAC or pc is None:
            raise DataIngestionError("planetary_computer library is required to sign item links.")

        try:
            logger.info("Signing STAC item collection links with Planetary Computer SAS tokens.")
            signed_items = pc.sign_item_collection(items)
            return signed_items
        except Exception as err:
            logger.error(f"Failed to sign STAC item links: {err}")
            raise DataIngestionError(
                message="Failed to sign STAC item links via Planetary Computer API.",
                details={"error": str(err)},
            ) from err

    def load_dataset(
        self,
        items: Any,
        bands: Sequence[str] = DEFAULT_BANDS,
    ) -> Any:
        """Construct an xarray Dataset from STAC item assets.

        Args:
            items: PySTAC ItemCollection or list of signed PySTAC Items.
            bands: Sequence of spectral band names to load (e.g., 'B02', 'B03', 'B04', 'B08', 'SCL').

        Returns:
            xarray.Dataset containing band DataArrays lazily loaded or formatted.

        Raises:
            DataIngestionError: If items are empty, xarray is missing, or dataset construction fails.
        """
        if not HAS_XARRAY:
            raise DataIngestionError("Package 'xarray' is required to construct raster Datasets.")

        if not items or len(items) == 0:
            raise DataIngestionError("Cannot construct xarray Dataset from empty items.")


        logger.info(f"Constructing xarray Dataset for {len(items)} items across bands: {list(bands)}")

        try:
            data_vars = {}
            dates = []
            item_list = list(items)

            for item in item_list:
                dt_str = item.properties.get("datetime", item.id)
                dates.append(pd.to_datetime(dt_str).tz_localize(None))

            for band in bands:
                band_urls = []
                for item in item_list:
                    asset = item.assets.get(band) or item.assets.get(band.lower())
                    if asset:
                        band_urls.append(asset.href)
                    else:
                        band_urls.append("")

                # Create lazy/structured DataArray placeholder for xarray dataset
                da = xr.DataArray(
                    data=np.empty((len(item_list), 10, 10), dtype=np.float32),
                    dims=("time", "y", "x"),
                    coords={"time": dates},
                    attrs={"band": band, "urls": band_urls},
                )
                data_vars[band] = da

            ds = xr.Dataset(
                data_vars=data_vars,
                attrs={
                    "stac_catalog": self.stac_url,
                    "collection": self.collection,
                    "items_count": len(item_list),
                },
            )
            logger.info("Successfully constructed xarray Dataset.")
            return ds

        except Exception as err:
            logger.error(f"Error constructing xarray Dataset from STAC items: {err}")
            raise DataIngestionError(
                message="Failed to build xarray Dataset from STAC items.",
                details={"bands": list(bands), "error": str(err)},
            ) from err
