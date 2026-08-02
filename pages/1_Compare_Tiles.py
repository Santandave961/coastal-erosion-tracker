import streamlit as st
import numpy as np

from src.preprocessing import load_s2_tile, compute_indices, shoreline_change, mangrove_change

st.set_page_config(page_title="Compare Tiles", page_icon="🛰️", layout="wide")
st.title("🛰️ Compare Before/After Sentinel-2 Tiles")
st.caption(
    "Upload two Sentinel-2 GeoTIFF tiles (Green, Red, NIR bands) from "
    "different dates over the same location to compute shoreline and "
    "mangrove canopy change."
)

col1, col2 = st.columns(2)
before_file = col1.file_uploader("Before tile (.tif)", type=["tif", "tiff"], key="before")
after_file = col2.file_uploader("After tile (.tif)", type=["tif", "tiff"], key="after")

years = st.number_input("Years between the two dates", value=1.0, min_value=0.1)

if before_file and after_file:
    before_path = f"/tmp/{before_file.name}"
    after_path = f"/tmp/{after_file.name}"
    with open(before_path, "wb") as f:
        f.write(before_file.getbuffer())
    with open(after_path, "wb") as f:
        f.write(after_file.getbuffer())

    with st.spinner("Computing NDVI/NDWI..."):
        before_tile = load_s2_tile(before_path)
        after_tile = load_s2_tile(after_path)

        before_idx = compute_indices(before_tile)
        after_idx = compute_indices(after_tile)

        erosion = shoreline_change(before_idx["ndwi"], after_idx["ndwi"])
        mangrove = mangrove_change(before_idx["ndvi"], after_idx["ndvi"])

    st.write("after_idx ndvi stats:",
             "min:", np.nanmin(after_idx["ndvi"]),
             "max:", np.nanmax(after_idx["ndvi"]),
             "NaN count:", np.isnan(after_idx["ndvi"]).sum())
    st.write("before_idx ndvi stats:",
             "min:", np.nanmin(before_idx["ndvi"]),
             "max:", np.nanmax(before_idx["ndvi"]),
             "NaN count:", np.isnan(before_idx["ndvi"]).sum())

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Net shoreline erosion", f"{erosion['net_erosion_pct']}%")
    c2.metric("Net canopy loss", f"{-mangrove['net_canopy_change_pct']}%")
    c3.metric("Annualized erosion", f"{erosion['net_erosion_pct'] / years:.2f}%/yr")

    st.divider()
    ic1, ic2 = st.columns(2)
    ic1.image(np.clip(before_idx["ndvi"], -1, 1), caption="NDVI — Before", clamp=True, use_column_width=True)
    ic2.image(np.clip(after_idx["ndvi"], -1, 1), caption="NDVI — After", clamp=True, use_column_width=True)

    ic3, ic4 = st.columns(2)
    ic3.image(np.clip(before_idx["ndwi"], -1, 1), caption="NDWI — Before", clamp=True, use_column_width=True)
    ic4.image(np.clip(after_idx["ndwi"], -1, 1), caption="NDWI — After", clamp=True, use_column_width=True)

    st.info(
        "Copy the erosion/canopy loss percentages above into the Home page "
        "form to add this zone to the risk map."
    )
else:
    st.info("Upload both tiles to compute change.")