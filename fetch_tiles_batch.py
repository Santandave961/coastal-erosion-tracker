"""
fetch_tiles_batch.py - Pull Sentinel-2 tiles across multiple Niger Delta coastal
points so you have enough examples to actually train the CNN.

Built on the `sentinelhub` package (matches the request.json / hashed-folder
caching pattern already showing up in your sentinel2_tiles/ directory -- that
folder naming is SentinelHubRequest's own cache, so this reuses the same
CACHE_FOLDER convention rather than reinventing the request format).

Requires credentials from Copernicus Dataspace (CDSE) in a .env file:
    SH_CLIENT_ID=your_client_id
    SH_CLIENT_SECRET=your_client_secret

Usage:
    python fetch_tiles_batch.py                          # fetch all locations, default date
    python fetch_tiles_batch.py --dates 2024-01-15 2025-01-15 2026-01-15
    python fetch_tiles_batch.py --locations locations.csv
"""

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
import os

from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    CRS,
    BBox,
    bbox_to_dimensions,
)

load_dotenv()

# --- Niger Delta coastal points -------------------------------------------------
# Approximate coordinates for known Niger Delta coastal / erosion-relevant sites.
# These are starting points, not surveyed erosion-hotspot boundaries -- verify /
# refine against your NOSDRA cross-referencing and adjust before final labeling.
DEFAULT_LOCATIONS = [
    {"name": "bonny_island",      "lat": 4.4400, "lon": 7.1700},
    {"name": "brass",             "lat": 4.3167, "lon": 6.2333},
    {"name": "forcados",          "lat": 5.3500, "lon": 5.3667},
    {"name": "escravos",          "lat": 5.5833, "lon": 5.1833},
    {"name": "nembe",             "lat": 4.5333, "lon": 6.4000},
    {"name": "koluama",           "lat": 4.6333, "lon": 5.9833},
    {"name": "opobo",             "lat": 4.5333, "lon": 7.5167},
    {"name": "ayetoro",           "lat": 6.1167, "lon": 4.6833},
    {"name": "awoye",             "lat": 6.0833, "lon": 4.6167},
    {"name": "akassa",            "lat": 4.3500, "lon": 6.0667},
    {"name": "san_bartholomew_bay", "lat": 4.4833, "lon": 5.8500},
    {"name": "digitari",          "lat": 4.5167, "lon": 6.1000},
]

BUFFER_DEG = 0.06  # ~6.6km half-width bbox around each point. Widened from 0.03
                    # because several corrected coastal coordinates (Ayetoro,
                    # Bonny Island) still missed the shoreline at the tighter
                    # buffer -- a wider box gives more margin for coordinate
                    # imprecision while still centering roughly on the target.
RESOLUTION = 10    # meters per pixel

EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: ["B04", "B03", "B02"],
        output: { bands: 3 }
    };
}
function evaluatePixel(sample) {
    return [sample.B04, sample.B03, sample.B02];
}
"""


def get_config():
    config = SHConfig()
    config.sh_client_id = os.environ.get("SH_CLIENT_ID")
    config.sh_client_secret = os.environ.get("SH_CLIENT_SECRET")
    config.sh_base_url = os.environ.get("SH_BASE_URL", "https://sh.dataspace.copernicus.eu")
    config.sh_token_url = os.environ.get(
        "SH_TOKEN_URL", "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    )
    if not config.sh_client_id or not config.sh_client_secret:
        raise SystemExit(
            "Missing SH_CLIENT_ID / SH_CLIENT_SECRET. Add them to your .env file "
            "(Copernicus Dataspace credentials)."
        )
    return config


def load_locations(csv_path: str = None):
    if not csv_path:
        return DEFAULT_LOCATIONS
    locations = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            locations.append({
                "name": row["name"].strip(),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
            })
    return locations


def fetch_tile(config, name, lat, lon, date, cache_dir, window_days=15):
    bbox = BBox(
        bbox=[lon - BUFFER_DEG, lat - BUFFER_DEG, lon + BUFFER_DEG, lat + BUFFER_DEG],
        crs=CRS.WGS84,
    )
    size = bbox_to_dimensions(bbox, resolution=RESOLUTION)

    # DataCollection.SENTINEL2_L2A defaults to services.sentinel-hub.com.
    # CDSE (sh.dataspace.copernicus.eu) needs the collection explicitly rebound
    # to that service URL, or every request 401s regardless of valid credentials.
    cdse_collection = DataCollection.SENTINEL2_L2A.define_from(
        "s2l2a_cdse", service_url=config.sh_base_url
    )

    # A single-day time_interval gives mosaicking_order="leastCC" nothing to
    # choose between -- it just returns whatever's available that exact day,
    # cloud or not. Widen to a +/- window so it can actually pick a clear scene.
    center = datetime.strptime(date, "%Y-%m-%d")
    start = (center - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (center + timedelta(days=window_days)).strftime("%Y-%m-%d")

    request = SentinelHubRequest(
        data_folder=str(cache_dir),
        evalscript=EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=cdse_collection,
                time_interval=(start, end),
                mosaicking_order="leastCC",
                maxcc=0.3,  # discard candidate scenes with >30% overall cloud cover
                             # before leastCC even picks among them
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )

    data = request.get_data(save_data=True)
    if not data:
        return None

    # SentinelHubRequest caches under data_folder/<hash>/response.tiff
    # find the most recently written response file for this request
    out_dirs = sorted(cache_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in out_dirs:
        tif = next(d.glob("response.tif*"), None)
        if tif:
            return tif
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--locations", help="CSV with columns: name,lat,lon (defaults to built-in Niger Delta list)")
    parser.add_argument("--dates", nargs="+", default=["2025-06-15"], help="One or more dates (YYYY-MM-DD)")
    parser.add_argument("--out", default="sentinel2_tiles", help="Cache/output directory")
    parser.add_argument("--manifest", default="tile_manifest.csv", help="CSV log of every tile fetched")
    parser.add_argument("--window_days", type=int, default=30,
                         help="Search +/- this many days around each date for a low-cloud scene")
    args = parser.parse_args()

    config = get_config()
    locations = load_locations(args.locations)
    cache_dir = Path(args.out)
    cache_dir.mkdir(exist_ok=True)

    manifest_rows = []
    total = len(locations) * len(args.dates)
    count = 0

    for loc in locations:
        for date in args.dates:
            count += 1
            print(f"[{count}/{total}] {loc['name']} @ {date} ...", end=" ")
            try:
                tif_path = fetch_tile(config, loc["name"], loc["lat"], loc["lon"], date, cache_dir, args.window_days)
                if tif_path:
                    print(f"OK -> {tif_path}")
                    manifest_rows.append({
                        "name": loc["name"], "lat": loc["lat"], "lon": loc["lon"],
                        "date": date, "path": str(tif_path.relative_to(cache_dir)),
                    })
                else:
                    print("FAILED (no data returned -- likely heavy cloud cover for this date)")
            except Exception as e:
                print(f"ERROR: {e}")

    manifest_path = Path(args.manifest)
    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "lat", "lon", "date", "path"])
        if write_header:
            writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone. {len(manifest_rows)}/{total} tiles fetched successfully.")
    print(f"Manifest written to: {manifest_path}")
    print("Next: run preprocess_tile.py --batch on the output folder, "
          "then label each tile in the manifest (eroded=1 / stable=0) to build labels.csv.")


if __name__ == "__main__":
    main()