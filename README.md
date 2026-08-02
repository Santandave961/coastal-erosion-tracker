# Coastal Erosion & Mangrove Loss Tracker

Satellite-based tracking of shoreline retreat and mangrove canopy loss
across the Niger Delta coastline, using Sentinel-2 NDVI/NDWI time-series
comparison. Served via **FastAPI**, visualized on a **Streamlit** map
dashboard.

## Business case

Niger Delta and coastal Nigeria are losing land to erosion and mangrove
degradation — driven by oil infrastructure, illegal sand mining, and
rising sea levels — but it's rarely tracked systematically at scale.

- **Coastal insurers & reinsurers** — price property/infrastructure risk
  in eroding zones based on actual measured retreat rates, not estimates.
- **Oil & gas companies** — environmental impact reporting tied to their
  own dredging/pipeline activity; liability exposure tracking.
- **World Bank / AfDB climate adaptation projects** — Nigeria's coastal
  resilience funding needs exactly this kind of independent monitoring
  data to prioritize spend.
- **Carbon credit verifiers** — mangroves are a blue carbon asset; loss
  tracking ties directly into carbon credit invalidation/verification.
- **State governments (Bayelsa, Delta, Lagos)** — coastal defense budget
  prioritization by measured erosion rate per LGA/community.

## How it works

1. **NDVI** (vegetation index) tracks mangrove canopy health/extent —
   canopy loss shows up as NDVI dropping below a vegetation threshold.
2. **NDWI** (water index) tracks water/land boundary — shoreline retreat
   shows up as NDWI rising above a water threshold in a previously-land
   pixel.
3. Compare two Sentinel-2 tiles (same location, different dates) to
   quantify % canopy lost and % land lost to water between the two dates.
4. A rule-based composite score (0-100) combines both signals into a
   single risk score per zone — no training data required, same pattern
   as the flood risk monitor. Swap in a trained model later once you have
   historical ground-truth erosion data.

## Project structure

```
coastal-erosion-tracker/
├── app.py                      # Streamlit dashboard (calls the API)
├── pages/
│   └── 1_Compare_Tiles.py      # Upload before/after tiles, compute change
├── api/
│   └── main.py                  # FastAPI endpoints
├── src/
│   ├── preprocessing.py         # NDVI/NDWI computation from S2 bands
│   ├── risk_model.py             # Composite erosion/canopy-loss scoring
│   └── map_utils.py              # Folium map builder
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── entrypoint.sh
```

## Getting Sentinel-2 tiles

Use the same Sentinel Hub Process API approach as the Niger Delta
Oil Spill & Gas Flare Monitor — request Green (B03), Red (B04), and NIR
(B08) bands for a bounding box and date, which returns an
already-georeferenced GeoTIFF ready to use with `src/preprocessing.py`.
(A `fetch_copernicus_tiles.py`-style script can be adapted from that
project — ask if you want this built out here too.)

## Running locally (without Docker)

```bash
pip install -r requirements.txt

# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Dashboard
streamlit run app.py
```

API docs: http://localhost:8000/docs

## Running with Docker

**Single container (simplest, good for demos):**
```bash
docker build -t coastal-erosion-tracker .
docker run -p 8000:8000 -p 8501:8501 coastal-erosion-tracker
```

**Two containers (API and dashboard scale independently):**
```bash
docker-compose up --build
```

## Known gaps / next steps

- No automated tile-fetching script yet (see "Getting Sentinel-2 tiles"
  above) — currently manual upload via the Compare Tiles page.
- Rule-based scoring hasn't been validated against real historical
  erosion/mangrove-loss ground truth — weights are a reasonable starting
  point, not calibrated.
- No cloud-masking on Sentinel-2 tiles yet — cloud cover will distort
  NDVI/NDWI readings; worth adding a cloud-probability band filter.
- No time-series view yet (only two-date comparison) — a fuller version
  would show a trend line across multiple years per zone.
