import streamlit as st
import pandas as pd
import requests
import os

from src.map_utils import build_erosion_map
from streamlit_folium import st_folium
from src.risk_model import score_zone

API_URL = os.getenv("EROSION_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Coastal Erosion & Mangrove Loss Tracker", page_icon="🌴", layout="wide")
st.title("🌴 Coastal Erosion & Mangrove Loss Tracker")
st.caption(
    "Sentinel-2 NDVI/NDWI time-series analysis — tracking shoreline retreat "
    "and mangrove canopy loss across the Niger Delta coastline, served via FastAPI."
)

st.sidebar.info("Running self-contained - no external API needed")

st.divider()
st.subheader("Score a coastal zone")

with st.form("zone_form"):
    col1, col2 = st.columns(2)
    zone_name = col1.text_input("Zone name", "Nembe Coastline")
    date_val = col2.date_input("Assessment date")

    col3, col4 = st.columns(2)
    lat = col3.number_input("Latitude", value=4.5, format="%.5f")
    lon = col4.number_input("Longitude", value=6.4, format="%.5f")

    col5, col6, col7 = st.columns(3)
    erosion_pct = col5.slider("Net shoreline erosion (%)", -20.0, 20.0, 2.0)
    canopy_loss_pct = col6.slider("Net mangrove canopy loss (%)", -20.0, 20.0, 3.0)
    years = col7.number_input("Years between comparison dates", value=1.0, min_value=0.1)

    submitted = st.form_submit_button("Score zone")

if submitted:
    payload = {
        "zone_name": zone_name,
        "latitude": lat,
        "longitude": lon,
        "net_erosion_pct": erosion_pct,
        "net_canopy_loss_pct": canopy_loss_pct,
        "years_observed": years,
        "assessment_date": str(date_val),
    }
    try:
        result = score_zone(
            net_erosion_pct=payload["net_erosion_pct"],
            net_canopy_loss_pct=payload["net_canopy_loss_pct"],
            years_observed=payload["years_observed"]
        )
        level_color = {"Severe": "🔴", "Moderate": "🟠", "Low": "🟢"}
        st.metric(
            "Risk Score",
            f"{result['risk_score']}/100",
            delta=f"{level_color.get(result['risk_level'], '')} {result['risk_level']} risk",
        )
        c1, c2 = st.columns(2)
        c1.metric("Erosion rate", f"{result['annualized_erosion_pct_per_year']}%/yr")
        c2.metric("Canopy loss rate", f"{result['annualized_canopy_loss_pct_per_year']}%/yr")

        existing = st.session_state.get("scored_zones", pd.DataFrame())
        new_row = pd.DataFrame([result])
        st.session_state["scored_zones"] = pd.concat([existing, new_row], ignore_index=True)

    except Exception as e:
        st.error(f"API call failed: {e}")

st.divider()
st.subheader("Scored zones map")

zones = st.session_state.get("scored_zones", pd.DataFrame())
if not zones.empty:
    m = build_erosion_map(zones)
    st_folium(m, width=None, height=500)
    st.dataframe(zones, use_container_width=True)
else:
    st.info("Score a zone above to see it appear on the map.")