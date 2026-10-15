"""Huit features de persistance : une anomalie subtile se voit a sa DUREE.

Un demarrage d'appareil dure 1 a 4 pas ; un biais de compteur dure 24 a 72.
Moyenner l'ecart sur plusieurs heures effondre le premier et conserve le
second. C'est ce qui separe ``falsified_reading`` (court) de ``sensor_bias``
(long), les deux se confondant sans cela.

Toutes les fenetres sont causales et groupees par maison.

Retenues par la campagne d'experiences : +0,087 de F1 sur ``sensor_bias``,
et taux de faux positifs divise par deux.
"""

from __future__ import annotations

import numpy as np

from .windows import rolling_mean, rolling_median, run_length

PERSISTENCE_FEATURE_NAMES: list[str] = [
    "mean_deviation_3h",
    "mean_deviation_6h",
    "mean_deviation_12h",
    "mean_deviation_24h",
    "median_res_zone_6h",
    "median_res_zone_24h",
    "persistent_neighbor_deviation",
    "consecutive_abnormal_count",
]

#: nombre de pas de 15 minutes par duree usuelle
STEPS_PER_DURATION = {"3h": 12, "6h": 24, "12h": 48, "24h": 96}


def build(builder, res_zone: np.ndarray,
          nb_neighbours: np.ndarray) -> dict[str, np.ndarray]:
    """Persistance de l'ecart, sur quatre echelles de temps.

    ``res_zone`` et ``nb_neighbours`` sont les features de base deja calculees :
    on les reutilise plutot que de les recalculer.
    """
    stats = builder.stats
    min_periods = int(builder.f["rolling_min_periods"])
    normalised = builder.normalised_power()
    z_normalised = (normalised - stats.norm_center) / np.maximum(
        stats.norm_scale, 1e-9)

    abnormal = np.abs(z_normalised) > stats.neighbour_z_threshold
    abnormal = np.where(np.isnan(z_normalised), False, abnormal)

    return {
        "mean_deviation_3h": rolling_mean(z_normalised,
                                          STEPS_PER_DURATION["3h"], min_periods),
        "mean_deviation_6h": rolling_mean(z_normalised,
                                          STEPS_PER_DURATION["6h"], min_periods),
        "mean_deviation_12h": rolling_mean(z_normalised,
                                           STEPS_PER_DURATION["12h"], min_periods),
        "mean_deviation_24h": rolling_mean(z_normalised,
                                           STEPS_PER_DURATION["24h"], min_periods),
        "median_res_zone_6h": rolling_median(res_zone,
                                             STEPS_PER_DURATION["6h"], min_periods),
        "median_res_zone_24h": rolling_median(res_zone,
                                              STEPS_PER_DURATION["24h"],
                                              min_periods),
        "persistent_neighbor_deviation": rolling_mean(
            nb_neighbours, STEPS_PER_DURATION["6h"], min_periods),
        "consecutive_abnormal_count": run_length(abnormal),
    }
