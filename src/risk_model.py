"""
Composite risk scoring for coastal erosion / mangrove loss, combining
shoreline retreat and canopy loss signals into a single score per zone.

Rule-based scoring (no training data needed) — same pattern as the flood
risk monitor. Once you have historical ground-truth erosion data, this can
be replaced with a trained model without changing the API/dashboard.
"""


def score_zone(net_erosion_pct: float, net_canopy_loss_pct: float,
               years_observed: float = 1.0) -> dict:
    """
    net_erosion_pct: positive = land lost to water (from shoreline_change)
    net_canopy_loss_pct: positive = mangrove canopy lost (from mangrove_change,
                          pass -net_canopy_change_pct since that fn returns
                          gain-loss, we want loss-gain)
    years_observed: time span between the two dates compared, for
                    normalizing an annualized rate
    """
    erosion_rate = max(net_erosion_pct, 0) / max(years_observed, 0.1)
    canopy_loss_rate = max(net_canopy_loss_pct, 0) / max(years_observed, 0.1)

    erosion_score = min(erosion_rate / 5.0, 1.0) * 50   # up to 50 pts (5%/yr = max)
    canopy_score = min(canopy_loss_rate / 5.0, 1.0) * 50  # up to 50 pts

    total = round(erosion_score + canopy_score, 1)
    if total >= 70:
        level = "Severe"
    elif total >= 40:
        level = "Moderate"
    else:
        level = "Low"

    return {
        "risk_score": total,
        "risk_level": level,
        "annualized_erosion_pct_per_year": round(erosion_rate, 2),
        "annualized_canopy_loss_pct_per_year": round(canopy_loss_rate, 2),
    }