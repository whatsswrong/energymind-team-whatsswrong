"""Les 12 features d'origine, derivees des seules mesures observables.

| # | Feature | Ce qu'elle mesure |
|---|---------------------|--------------------------------------------|
| 1 | res_zone            | incoherence maisons <-> compteur de zone   |
| 2 | res_trans           | incoherence zones <-> compteur du poste    |
| 3 | res_energy          | coherence E(t)-E(t-1) contre P x 0,25      |
| 4 | ratio_voisins       | la maison face aux autres de sa zone       |
| 5 | nb_voisins_anormaux | combien de voisins devient en meme temps   |
| 6 | variation           | variation de P normalisee par la baseline  |
| 7 | frozen_count        | pas consecutifs quasi identiques           |
| 8 | rolling_std_6h      | variabilite des 6 heures passees           |
| 9 | drift_slope         | pente de derive sur 12 heures              |
| 10| dev_tension         | ecart relatif de tension au nominal        |
| 11| trans_load_pct      | charge du transformateur en % du nominal   |
| 12| res_VI              | coherence P ~ V x I x PF / 1000            |

``res_zone`` et ``res_trans`` retranchent la PERTE TECHNIQUE ATTENDUE, modelisee
affine en charge : ils mesurent un ecart a la normale, pas la perte elle-meme.
"""

from __future__ import annotations

import numpy as np

from .windows import neighbour_median, rolling_slope, rolling_std, run_length

BASE_FEATURE_NAMES: list[str] = [
    "res_zone",
    "res_trans",
    "res_energy",
    "ratio_voisins",
    "nb_voisins_anormaux",
    "variation",
    "frozen_count",
    "rolling_std_6h",
    "drift_slope",
    "dev_tension",
    "trans_load_pct",
    "res_VI",
]


def build(builder) -> dict[str, np.ndarray]:
    """Calcule les 12 features de base en representation large ``(T, H)``.

    ``builder`` porte la vue large du reseau et les statistiques apprises sur
    le train (baselines, echelles robustes, seuils). Aucune valeur future n'est
    utilisee.
    """
    wide, raw = builder.wide, builder.raw_signals
    stats = builder.stats
    eps = float(builder.f["eps_power_kw"])
    eps_ratio = float(builder.f["eps_ratio"])
    zone_of_house = wide.zone_of_house

    # 1 / 2 : residus de niveau, centres-reduits avec les statistiques du train
    zone_centre, zone_scale = stats.zone_resid
    trans_centre, trans_scale = stats.trans_resid
    res_zone = ((builder.zone_residual() - zone_centre)
                / zone_scale)[:, zone_of_house]
    res_trans = np.repeat(((builder.trans_residual() - trans_centre)
                           / trans_scale)[:, None], wide.H, axis=1)

    # 3 : coherence energetique (kWh) — une difference, jamais l'index brut
    res_energy = raw["dE"] - wide.P * builder.dt

    # 4 / 5 : comparaison aux voisins, apres normalisation par la baseline
    normalised = builder.normalised_power()
    peer_median = neighbour_median(normalised, wide)
    ratio_voisins = normalised / np.maximum(peer_median, eps_ratio)

    z_normalised = (normalised - stats.norm_center) / np.maximum(
        stats.norm_scale, 1e-9)
    abnormal = np.abs(z_normalised) > stats.neighbour_z_threshold
    nb_voisins = np.zeros((wide.T, wide.H))
    for house in range(wide.H):
        peers = wide.neighbours(house)
        if peers.size:
            nb_voisins[:, house] = np.nansum(abnormal[:, peers], axis=1)

    # 6 : variation normalisee par la baseline de la maison
    baseline_now = stats.baseline[wide.slot, wide.is_weekend.astype(int)]
    variation = raw["dP"] / np.maximum(baseline_now, eps)

    # 7 : compteur fige, avec tolerance — jamais une egalite flottante stricte
    rel_dP = np.abs(raw["dP"]) / np.maximum(np.abs(wide.P), eps)
    rel_dV = np.abs(raw["dV"]) / np.maximum(np.abs(wide.V), 1.0)
    frozen_flag = (rel_dP <= stats.frozen_tol_P) & (rel_dV <= stats.frozen_tol_V)
    frozen_flag = np.where(np.isnan(rel_dP) | np.isnan(rel_dV), False,
                           frozen_flag)
    frozen_count = run_length(frozen_flag)

    # 8 : variabilite sur les 6 heures passees
    rolling_std_6h = rolling_std(wide.P,
                                 int(builder.f["rolling_std_window"]),
                                 int(builder.f["rolling_min_periods"]))

    # 9 : derive sur 12 h, mesuree sur l'ecart aux voisins — une hausse
    #     generale de la zone n'est donc pas vue comme une derive locale
    drift_signal = normalised - peer_median
    drift_slope = rolling_slope(drift_signal, int(builder.f["drift_window"]),
                                int(builder.f["drift_min_periods"]))

    # 10 / 11 / 12 : grandeurs instantanees
    dev_tension = (wide.V - builder.nominal_v) / builder.nominal_v
    trans_load_pct = np.repeat(
        (100.0 * wide.trans_P / builder.capacity)[:, None], wide.H, axis=1)
    res_VI = ((wide.P - wide.V * wide.I * wide.PF / 1000.0)
              / np.maximum(np.abs(wide.P), eps))

    features = {
        "res_zone": res_zone,
        "res_trans": res_trans,
        "res_energy": res_energy,
        "ratio_voisins": ratio_voisins,
        "nb_voisins_anormaux": nb_voisins,
        "variation": variation,
        "frozen_count": frozen_count,
        "rolling_std_6h": rolling_std_6h,
        "drift_slope": drift_slope,
        "dev_tension": dev_tension,
        "trans_load_pct": trans_load_pct,
        "res_VI": res_VI,
    }
    assert list(features) == BASE_FEATURE_NAMES, "ordre des features de base"
    return features
