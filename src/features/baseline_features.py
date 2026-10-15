"""Quatre features d'ecart a l'historique propre de la maison.

``ratio_voisins`` compare la maison a ses voisins : si toute la zone bouge, il
ne voit rien. Un biais de compteur, lui, decale la maison par rapport a son
PROPRE historique. Ces quatre vues capturent ce decalage.

Toutes les medianes de reference sont apprises sur le TRAIN uniquement. Ce sont
des medianes, non des moyennes : un demarrage d'appareil ne deplace pas une
mediane.

Retenues par la campagne d'experiences : +0,051 de F1 sur ``sensor_bias``.
"""

from __future__ import annotations

import numpy as np

BASELINE_FEATURE_NAMES: list[str] = [
    "deviation_from_house_baseline",
    "relative_deviation_from_baseline",
    "zscore_vs_house_baseline",
    "ratio_to_hour_median",
]


def build(builder) -> dict[str, np.ndarray]:
    """Ecart de chaque maison a son habitude, sous quatre formes."""
    wide, stats = builder.wide, builder.stats
    eps = float(builder.f["eps_power_kw"])

    baseline = stats.baseline[wide.slot, wide.is_weekend.astype(int)]
    normalised = builder.normalised_power()

    # mediane par (maison, heure) : plus grossiere que le creneau de 15 min,
    # donc plus stable quand l'historique disponible est court
    minutes_per_step = int(builder.cfg.get("grid.interval_minutes"))
    hour = wide.slot * minutes_per_step // 60
    hour_median = stats.hour_median[hour]

    return {
        "deviation_from_house_baseline": wide.P - baseline,
        "relative_deviation_from_baseline": wide.P / np.maximum(baseline, eps) - 1.0,
        "zscore_vs_house_baseline": (normalised - stats.norm_center)
        / np.maximum(stats.norm_scale, 1e-9),
        "ratio_to_hour_median": wide.P / np.maximum(hour_median, eps),
    }
