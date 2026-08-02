"""
preprocess_tile.py - Convert a raw Sentinel tile (.tif/.tiff) into a CNN-ready
normalized HWC numpy array (.npy).

Steps:
  1. Load tile as (bands, H, W)
  2. Clip to 1st/99th percentile per band (kills outlier hot pixels / speckle)
  3. Min-max scale each band to [0, 1]
  4. Transpose to (H, W, bands) -- standard CNN input layout
  5. Save as .npy next to the source file (or to --out)

Usage:
    python preprocess_tile.py path/to/response.tiff
    python preprocess_tile.py path/to/response.tiff --out preprocessed/tile_0001.npy
    python preprocess_tile.py sentinel2_tiles/  --batch   # process every .tif/.tiff in a folder
"""

import sys
import argparse
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError:
    sys.exit("rasterio not installed. Run: pip install rasterio")


def preprocess_tile(path: str, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)  # (bands, H, W)

    out_bands = []
    for band in data:
        valid = band[~np.isnan(band)]
        if valid.size == 0:
            out_bands.append(np.zeros_like(band))
            continue

        lo, hi = np.percentile(valid, [low_pct, high_pct])
        band_clipped = np.clip(band, lo, hi)

        if hi - lo < 1e-6:
            band_scaled = np.zeros_like(band)
        else:
            band_scaled = (band_clipped - lo) / (hi - lo)

        band_scaled = np.nan_to_num(band_scaled, nan=0.0)
        out_bands.append(band_scaled.astype(np.float32))

    stacked = np.stack(out_bands, axis=0)      # (bands, H, W)
    hwc = np.transpose(stacked, (1, 2, 0))     # (H, W, bands)
    return hwc


def process_one(path: Path, out_path: Path = None):
    arr = preprocess_tile(str(path))
    out_path = out_path or path.with_suffix(".npy")
    np.save(out_path, arr)
    print(f"{path.name} -> {out_path}  shape={arr.shape}  "
          f"min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to a .tif/.tiff file, or a folder if --batch is used")
    parser.add_argument("--out", help="Output .npy path (single-file mode only)")
    parser.add_argument("--batch", action="store_true", help="Process every .tif/.tiff found recursively under path")
    args = parser.parse_args()

    src = Path(args.path)

    if args.batch:
        if not src.is_dir():
            sys.exit(f"--batch requires a directory, got: {src}")
        tiles = list(src.rglob("*.tif")) + list(src.rglob("*.tiff"))
        if not tiles:
            sys.exit(f"No .tif/.tiff files found under {src}")
        print(f"Found {len(tiles)} tile(s).")
        for t in tiles:
            try:
                process_one(t)
            except Exception as e:
                print(f"  FAILED: {t} -> {e}")
    else:
        if not src.is_file():
            sys.exit(f"File not found: {src}")
        out = Path(args.out) if args.out else None
        process_one(src, out)


if __name__ == "__main__":
    main()