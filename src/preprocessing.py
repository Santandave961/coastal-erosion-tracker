"""
Preprocessing utilities for coastal erosion / mangrove loss detection using
Sentinel-2 multispectral bands.

- NDVI (Normalized Difference Vegetation Index): tracks mangrove canopy health/loss
  NDVI = (NIR - Red) / (NIR + Red)
- NDWI (Normalized Difference Water Index): tracks shoreline/water extent
  NDWI = (Green - NIR) / (Green + NIR)

Expected input: a Sentinel-2 GeoTIFF with at least Green, Red, NIR bands
(B03, B04, B08 in Sentinel-2 naming), stacked in that order.
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling

TARGET_SIZE = (256, 256)


def load_s2_tile(filepath: str) -> np.ndarray:
    """Load a Sentinel-2 GeoTIFF tile with Green, Red, NIR bands (in that order)."""
    with rasterio.open(filepath) as src:
        arr = src.read(
            out_shape=(src.count, *TARGET_SIZE),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
    return arr


def compute_ndvi(green: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = nir + red
    denom[denom == 0] = 1e-6
    return (nir - red) / denom


def compute_ndwi(green: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = green + nir
    denom[denom == 0] = 1e-6
    return (green - nir) / denom


def compute_indices(tile: np.ndarray) -> dict:
    """tile expected shape: (3, H, W) = [Green, Red, NIR]"""
    green, red, nir = tile[0], tile[1], tile[2]
    return {
        "ndvi": compute_ndvi(green, red, nir),
        "ndwi": compute_ndwi(green, red, nir),
    }


def shoreline_change(ndwi_before: np.ndarray, ndwi_after: np.ndarray,
                      water_threshold: float = 0.0) -> dict:
    """
    Compare water extent between two dates. Positive water_gain_pct means
    the shoreline retreated (land became water) -> erosion signal.
    """
    water_before = (ndwi_before > water_threshold).astype(np.uint8)
    water_after = (ndwi_after > water_threshold).astype(np.uint8)

    total_px = water_before.size
    water_gain_px = np.logical_and(water_after == 1, water_before == 0).sum()
    water_loss_px = np.logical_and(water_after == 0, water_before == 1).sum()

    return {
        "water_gain_pct": round(100 * water_gain_px / total_px, 2),
        "water_loss_pct": round(100 * water_loss_px / total_px, 2),
        "net_erosion_pct": round(100 * (water_gain_px - water_loss_px) / total_px, 2),
    }


def mangrove_change(ndvi_before: np.ndarray, ndvi_after: np.ndarray,
                     vegetation_threshold: float = 0.3) -> dict:
    """
    Compare mangrove/vegetation extent between two dates. Positive
    canopy_loss_pct means mangrove canopy was lost.
    """
    veg_before = (ndvi_before > vegetation_threshold).astype(np.uint8)
    veg_after = (ndvi_after > vegetation_threshold).astype(np.uint8)

    total_px = veg_before.size
    canopy_loss_px = np.logical_and(veg_after == 0, veg_before == 1).sum()
    canopy_gain_px = np.logical_and(veg_after == 1, veg_before == 0).sum()

    return {
        "canopy_loss_pct": round(100 * canopy_loss_px / total_px, 2),
        "canopy_gain_pct": round(100 * canopy_gain_px / total_px, 2),
        "net_canopy_change_pct": round(100 * (canopy_gain_px - canopy_loss_px) / total_px, 2),
    }