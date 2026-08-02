"""Folium map builders for the coastal erosion dashboard."""

import folium
import pandas as pd

NIGER_DELTA_COAST_CENTER = (4.4, 6.0)
DEFAULT_ZOOM = 9

RISK_COLORS = {"Severe": "red", "Moderate": "orange", "Low": "green"}


def build_erosion_map(zones: pd.DataFrame = None) -> folium.Map:
    m = folium.Map(location=NIGER_DELTA_COAST_CENTER, zoom_start=DEFAULT_ZOOM,
                    tiles="cartodbpositron")

    if zones is not None and not zones.empty:
        for _, row in zones.iterrows():
            color = RISK_COLORS.get(row.get("risk_level", "Low"), "blue")
            folium.CircleMarker(
                location=(row["latitude"], row["longitude"]),
                radius=10,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=_zone_popup(row),
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def _zone_popup(row) -> str:
    return f"""
    <b>{row.get('zone_name', 'Zone')}</b><br>
    Risk Level: {row.get('risk_level')}<br>
    Score: {row.get('risk_score')}/100<br>
    Erosion: {row.get('annualized_erosion_pct_per_year')}%/yr<br>
    Canopy Loss: {row.get('annualized_canopy_loss_pct_per_year')}%/yr
    """