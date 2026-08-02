"""
Generates two synthetic 3-band (Green, Red, NIR) GeoTIFF tiles
for testing the Compare Tiles page locally.

Run this INSIDE your coastal-erosion-tracker project folder:
    python make_test_tiles.py

Requires: rasterio, numpy (already in your requirements.txt)
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin

WIDTH, HEIGHT = 256, 256
# Rough Nembe coastline bounding box (adjust if needed)
transform = from_origin(6.35, 4.55, 0.0001, 0.0001)  # top-left lon/lat, pixel size

def make_tile(path, seed, shoreline_shift=0, mangrove_loss=0):
    rng = np.random.default_rng(seed)

    # Base synthetic bands: land (higher NIR) vs water (higher green/red ratio)
    green = rng.normal(0.15, 0.03, (HEIGHT, WIDTH)).astype("float32")
    red = rng.normal(0.12, 0.03, (HEIGHT, WIDTH)).astype("float32")
    nir = rng.normal(0.35, 0.05, (HEIGHT, WIDTH)).astype("float32")

    # Simulate a coastline: bottom rows = water (low NIR), top = land (high NIR)
    shore_row = HEIGHT // 2 + shoreline_shift
    nir[shore_row:, :] -= 0.25   # water has low NIR
    green[shore_row:, :] += 0.05  # water has slightly higher green reflectance

    # Simulate mangrove loss: reduce NIR in a patch near the shoreline
    if mangrove_loss:
        patch = slice(max(0, shore_row - 40), shore_row)
        nir[patch, 50:150] -= mangrove_loss

    stacked = np.clip(np.stack([green, red, nir]), 0, 1).astype("float32")

    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=HEIGHT, width=WIDTH,
        count=3, dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(stacked)
        dst.descriptions = ("Green", "Red", "NIR")

    print(f"Wrote {path}")

make_tile("test_before.tif", seed=1, shoreline_shift=0, mangrove_loss=0)
make_tile("test_after.tif", seed=2, shoreline_shift=15, mangrove_loss=0.15)

print("\nDone. Upload test_before.tif and test_after.tif to the Compare Tiles page.")
print("Expect: shoreline retreated ~15 rows, mangrove canopy loss visible near shore.")