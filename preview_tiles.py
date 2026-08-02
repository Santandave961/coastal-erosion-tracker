"""
preview_tiles.py - Build a single labeled contact-sheet image of every tile in
sentinel2_tiles/ so you can visually scan for erosion vs stable coastline
before filling in labels.csv.

Uses tile_manifest.csv (if present) to label each thumbnail with its location
name + date. Falls back to the folder hash if no manifest entry matches.

Usage:
    python preview_tiles.py
    python preview_tiles.py --tiles_dir sentinel2_tiles --manifest tile_manifest.csv --out preview.png
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import rasterio
except ImportError:
    raise SystemExit("rasterio not installed. Run: pip install rasterio")


def load_rgb_stretched(path, low_pct=2, high_pct=98):
    """Load a tile and percentile-stretch each band to 0-1 for viewable contrast."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)  # (bands, H, W)

    bands = []
    for band in data[:3]:  # only need first 3 (R,G,B order from the evalscript)
        valid = band[~np.isnan(band)]
        if valid.size == 0:
            bands.append(np.zeros_like(band))
            continue
        lo, hi = np.percentile(valid, [low_pct, high_pct])
        if hi - lo < 1e-6:
            bands.append(np.zeros_like(band))
        else:
            stretched = np.clip((band - lo) / (hi - lo), 0, 1)
            bands.append(np.nan_to_num(stretched))

    rgb = np.stack(bands, axis=-1)  # (H, W, 3)
    return rgb


def load_manifest_labels(manifest_path):
    """Map folder-hash -> 'name @ date' label, by matching the path column."""
    lookup = {}
    if not manifest_path or not Path(manifest_path).exists():
        return lookup
    with open(manifest_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            # path column looks like: <hash>/response.tiff or <hash>\response.tiff
            folder = row["path"].replace("\\", "/").split("/")[0]
            lookup[folder] = f"{row['name']} @ {row['date']}"
    return lookup


def load_raw_rgb(path):
    """Raw (unstretched) reflectance values -- needed for cloud scoring so
    thresholds mean the same thing across every tile, regardless of how dark
    or bright that specific tile's own pixel range happens to be."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)  # (bands, H, W)
    return np.transpose(data[:3], (1, 2, 0))  # (H, W, 3) raw


def estimate_cloud_and_nodata(raw_rgb, cloud_reflectance_thresh=0.25, band_diff_thresh=0.08,
                               nodata_thresh=0.005):
    """Score on RAW reflectance, not the per-tile percentile-stretched display image.
    Percentile stretching is relative to each tile's own min/max, so a fixed
    'brightness > X' cutoff means something different on every tile -- that was
    why cloud-heavy tiles were scoring as clean. Fixed absolute thresholds on
    raw values fix that.
    """
    brightness = raw_rgb.mean(axis=-1)
    band_range = raw_rgb.max(axis=-1) - raw_rgb.min(axis=-1)

    nodata_like = brightness < nodata_thresh
    cloud_like = (brightness > cloud_reflectance_thresh) & (band_range < band_diff_thresh) & (~nodata_like)

    return float(cloud_like.mean()), float(nodata_like.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_dir", default="sentinel2_tiles")
    parser.add_argument("--manifest", default="tile_manifest.csv")
    parser.add_argument("--out", default="preview_contact_sheet.png")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--sort_by_clarity", action="store_true",
                         help="Sort tiles cleanest-first instead of alphabetically")
    parser.add_argument("--score_csv", default="tile_scores.csv",
                         help="Also write all cloud/nodata scores to this CSV for downstream use")
    args = parser.parse_args()

    tiles_dir = Path(args.tiles_dir)
    tif_paths = sorted(list(tiles_dir.rglob("response.tif")) + list(tiles_dir.rglob("response.tiff")))

    if not tif_paths:
        raise SystemExit(f"No response.tif/.tiff files found under {tiles_dir}")

    print(f"Found {len(tif_paths)} tile(s). Scoring cloud cover...")

    labels = load_manifest_labels(args.manifest)

    scored = []
    for tif_path in tif_paths:
        folder = tif_path.parent.name
        label = labels.get(folder, folder[:12])
        try:
            raw_rgb = load_raw_rgb(tif_path)
            cloud_frac, nodata_frac = estimate_cloud_and_nodata(raw_rgb)
            display_rgb = load_rgb_stretched(tif_path)
            bad_frac = cloud_frac + nodata_frac  # either one makes the tile unusable
        except Exception:
            display_rgb = None
            cloud_frac, nodata_frac, bad_frac = 1.0, 0.0, 1.0
        scored.append((tif_path, label, display_rgb, cloud_frac, nodata_frac, bad_frac))

    if args.sort_by_clarity:
        scored.sort(key=lambda x: x[5])

    print("\nCloud+nodata fraction estimate (lower = cleaner):")
    for _, label, _, cloud_frac, nodata_frac, bad_frac in sorted(scored, key=lambda x: x[5]):
        flag = "CLEAN" if bad_frac < 0.15 else ("PARTIAL" if bad_frac < 0.5 else "CLOUDY")
        print(f"  bad={bad_frac:.2f} (cloud={cloud_frac:.2f} nodata={nodata_frac:.2f})  [{flag:7s}]  {label}")

    with open(args.score_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "label", "cloud_frac", "nodata_frac", "bad_frac", "flag"])
        for tif_path, label, _, cloud_frac, nodata_frac, bad_frac in scored:
            flag = "CLEAN" if bad_frac < 0.15 else ("PARTIAL" if bad_frac < 0.5 else "CLOUDY")
            writer.writerow([tif_path.parent.name, label, f"{cloud_frac:.3f}", f"{nodata_frac:.3f}",
                              f"{bad_frac:.3f}", flag])
    print(f"Scores written to: {args.score_csv}")

    n = len(scored)
    cols = args.cols
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.5))
    axes = np.array(axes).reshape(-1)

    for i, (tif_path, label, display_rgb, cloud_frac, nodata_frac, bad_frac) in enumerate(scored):
        if display_rgb is not None:
            axes[i].imshow(display_rgb)
        else:
            axes[i].text(0.5, 0.5, "FAILED", ha="center", va="center", fontsize=8)
        axes[i].set_title(f"{label}\nbad~{bad_frac:.0%}", fontsize=8)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved contact sheet -> {args.out}")


if __name__ == "__main__":
    main()