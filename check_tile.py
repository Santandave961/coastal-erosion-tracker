"""
check_tile.py - Sanity-check a downloaded Sentinel tile before it enters preprocessing.

Catches:
  - Blank/black tiles (all-zero or near-zero pixels)
  - Fully saturated tiles (all-white / max value)
  - Heavy cloud cover (very high mean + low std, i.e. flat bright)
  - NaN / nodata-dominated tiles
  - Suspiciously low variance across bands

Usage:
    python check_tile.py path/to/tile.tif
    python check_tile.py sentinel2_tiles/7456ebadd297da9ff0eb53c461892332/response.tif
"""

import sys
import numpy as np

try:
    import rasterio
except ImportError:
    sys.exit("rasterio not installed. Run: pip install rasterio")


def check_tile(path: str, cloud_mean_thresh: float = 0.85, cloud_std_thresh: float = 0.05,
               blank_mean_thresh: float = 0.02, nodata_frac_thresh: float = 0.3):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)  # shape: (bands, height, width)
        nodata = src.nodata

    print(f"File: {path}")
    print(f"Shape: {data.shape} (bands, height, width)")
    print(f"Dtype (on disk): {src.dtypes if hasattr(src, 'dtypes') else 'n/a'}")

    # Normalize to 0-1 for thresholding, based on the actual max in the data
    data_max = np.nanmax(data) if np.nanmax(data) > 0 else 1.0
    norm = data / data_max

    issues = []

    # NaN / nodata check
    nan_frac = np.isnan(data).mean()
    nodata_frac = (data == nodata).mean() if nodata is not None else 0.0
    if nan_frac > nodata_frac_thresh:
        issues.append(f"High NaN fraction: {nan_frac:.1%}")
    if nodata_frac > nodata_frac_thresh:
        issues.append(f"High nodata fraction: {nodata_frac:.1%}")

    # Per-band stats
    print("\nPer-band stats:")
    for i, band in enumerate(norm, start=1):
        valid = band[~np.isnan(band)]
        if valid.size == 0:
            print(f"  Band {i}: EMPTY (all NaN)")
            issues.append(f"Band {i} is entirely NaN")
            continue
        b_mean, b_std = valid.mean(), valid.std()
        print(f"  Band {i}: mean={b_mean:.3f}  std={b_std:.4f}  min={valid.min():.3f}  max={valid.max():.3f}")

        if b_mean < blank_mean_thresh and b_std < 0.01:
            issues.append(f"Band {i} looks blank/black (mean={b_mean:.3f}, std={b_std:.4f})")
        if b_mean > cloud_mean_thresh and b_std < cloud_std_thresh:
            issues.append(f"Band {i} looks cloud-saturated (mean={b_mean:.3f}, std={b_std:.4f})")

    print("\n--- Verdict ---")
    if issues:
        print("FLAGGED — likely unusable tile:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("OK — tile passes basic sanity checks.")
        return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python check_tile.py path/to/tile.tif")
    ok = check_tile(sys.argv[1])
    sys.exit(0 if ok else 1)