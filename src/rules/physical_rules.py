"""Moteur de regles de coherence physique / reseau.

Ce n'est pas un modele : c'est une poignee de regles lisibles qui exploitent la
**redondance** entre niveaux de mesure. Principe directeur : aucun capteur
n'est une verite absolue.

Trois agregats sont compares, choisis pour etre mutuellement independants :

    A  somme des compteurs de maison d'une zone
    B  compteur de la zone
    C  compteur du transformateur

``zone_rel`` compare A et B. ``trans_vs_zones_rel`` compare B et C.
``trans_vs_houses_rel`` compare A et C — c'est lui qui permet de trancher,
car il ne depend d'aucun compteur de zone.

    CASE 1  une maison incoherente, voisins et niveaux superieurs normaux
            -> anomalie locale (maison ou son compteur)
    CASE 2  plusieurs maisons bougent ensemble et la zone suit
            -> evenement reseau reel
    CASE 3  A et C s'accordent, B s'ecarte
            -> defaut du compteur de ZONE
    CASE 4  A et B s'accordent entre eux, C s'ecarte des deux
            -> defaut du compteur du TRANSFORMATEUR
    CASE 5  preuves contradictoires entre niveaux
            -> ambigu

Tous les seuils sont des quantiles estimes sur les lignes NORMALES du TRAIN :
aucune constante magique, aucun reglage sur le test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

VERDICT_NONE = "none"
VERDICT_LOCAL = "local_house_anomaly"
VERDICT_GRID = "grid_event"
VERDICT_ZONE_METER = "zone_meter_fault"
VERDICT_TRANS_METER = "transformer_meter_fault"
VERDICT_AMBIGUOUS = "ambiguous"

#: verdicts qui constituent une preuve d'anomalie au niveau maison
ANOMALY_VERDICTS = (VERDICT_LOCAL, VERDICT_GRID)

#: colonnes de redondance attendues (produites par FeatureBuilder.rule_signals)
SIGNAL_COLUMNS = ("zone_rel", "zone_v_rel", "trans_vs_zones_rel",
                  "trans_vs_houses_rel")


@dataclass
class RuleThresholds:
    """Seuils appris sur le train normal (quantile configure)."""

    zone_rel: float = 0.03
    zone_v_rel: float = 0.02
    trans_vs_zones_rel: float = 0.02
    trans_vs_houses_rel: float = 0.03
    res_vi: float = 0.05
    res_energy: float = 0.05
    ratio: float = 0.5
    min_coherent_neighbours: int = 2
    quantile: float = 0.995
    #: perte technique attendue (mediane du train propre) retranchee avant
    #: comparaison : on mesure l'ecart a la normale, pas l'ecart a zero
    centers: dict = field(default_factory=dict)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"threshold": name, "value": round(float(getattr(self, name)), 6)}
            for name in ("zone_rel", "zone_v_rel", "trans_vs_zones_rel",
                         "trans_vs_houses_rel",
                         "res_vi", "res_energy", "ratio",
                         "min_coherent_neighbours", "quantile")])


def fit_thresholds(X: pd.DataFrame, signals: pd.DataFrame,
                   clean_rows: np.ndarray, global_clean_rows: np.ndarray,
                   train_rows: np.ndarray, cfg) -> RuleThresholds:
    """Estime chaque seuil comme un quantile haut sur le TRAIN PROPRE.

    ``clean_rows`` marque les instants ou aucun niveau n'est perturbe : c'est
    la seule reference valable, un voisin en panne suffisant a fausser le
    residu de zone d'une maison pourtant normale.
    """
    q = float(cfg.get("rules.threshold_quantile"))
    mask = train_rows & clean_rows
    # les signaux qui dependent de TOUT le reseau (transformateur) exigent un
    # instant ou aucune zone n'est perturbee, pas seulement celle de la ligne
    mask_global = train_rows & global_clean_rows
    assert mask.sum() > 1000, "trop peu de lignes propres pour calibrer les regles"
    assert mask_global.sum() > 1000, "trop peu d'instants entierement propres"

    def quant(frame: pd.DataFrame, col: str, transform=np.abs,
              floor: float = 1e-6, rows=None) -> float:
        rows = mask if rows is None else rows
        values = transform(frame.loc[rows, col].to_numpy(dtype=float))
        return max(float(np.nanquantile(values, q)), floor)

    centers = {col: float(np.nanmedian(
        signals.loc[mask_global if "trans" in col else mask, col]))
        for col in SIGNAL_COLUMNS}

    def centred(col: str, rows=None) -> float:
        return quant(signals, col, lambda v: np.abs(v - centers[col]), rows=rows)

    return RuleThresholds(
        centers=centers,
        zone_rel=centred("zone_rel"),
        zone_v_rel=centred("zone_v_rel"),
        trans_vs_zones_rel=centred("trans_vs_zones_rel", rows=mask_global),
        trans_vs_houses_rel=centred("trans_vs_houses_rel", rows=mask_global),
        res_vi=quant(X, "res_VI"),
        res_energy=quant(X, "res_energy"),
        ratio=quant(X, "ratio_voisins", lambda v: np.abs(v - 1.0)),
        min_coherent_neighbours=int(cfg.get("rules.min_coherent_neighbors")),
        quantile=q,
    )


def _exceeds(values: np.ndarray, limit: float) -> np.ndarray:
    """Depassement de seuil ; une mesure absente compte comme une anomalie."""
    return np.where(np.isnan(values), True, np.abs(values) > limit)


def evidence_frame(X: pd.DataFrame, signals: pd.DataFrame,
                   t: RuleThresholds) -> pd.DataFrame:
    """Les cinq preuves elementaires, avant application des cas."""
    return pd.DataFrame({
        "house_internal": (_exceeds(X["res_VI"].to_numpy(float), t.res_vi)
                           | _exceeds(X["res_energy"].to_numpy(float),
                                      t.res_energy)),
        # ecart de la maison a son propre comportement habituel, mesure en
        # ecarts-types robustes : un seuil sur le ratio brut serait trop lache
        # (les demarrages d'appareils font deja x3) et laisserait passer une
        # maison tombee a zero
        "house_vs_peers": (signals["house_deviates"].to_numpy(dtype=bool)
                           | _exceeds(X["ratio_voisins"].to_numpy(float) - 1.0,
                                      t.ratio)),
        # le compteur de zone peut mentir sur la puissance OU sur la tension
        "zone_mismatch": (
            _exceeds(signals["zone_rel"].to_numpy(float)
                     - t.centers.get("zone_rel", 0.0), t.zone_rel)
            | _exceeds(signals["zone_v_rel"].to_numpy(float)
                       - t.centers.get("zone_v_rel", 0.0), t.zone_v_rel)),
        "trans_vs_zones_mismatch": _exceeds(
            signals["trans_vs_zones_rel"].to_numpy(float)
            - t.centers.get("trans_vs_zones_rel", 0.0), t.trans_vs_zones_rel),
        "trans_vs_houses_mismatch": _exceeds(
            signals["trans_vs_houses_rel"].to_numpy(float)
            - t.centers.get("trans_vs_houses_rel", 0.0), t.trans_vs_houses_rel),
        "peers_moving": (np.nan_to_num(
            X["nb_voisins_anormaux"].to_numpy(float), nan=0.0)
            >= t.min_coherent_neighbours),
    }, index=X.index)


def apply_rules(X: pd.DataFrame, signals: pd.DataFrame,
                thresholds: RuleThresholds,
                zone_key: np.ndarray) -> pd.DataFrame:
    """Verdict par observation maison + preuves intermediaires.

    ``zone_key`` identifie le couple (timestamp, zone) : il permet de compter
    combien de maisons de la zone s'ecartent au meme instant, ce qui est le
    discriminant entre "le compteur de zone ment" et "un compteur de maison
    ment" — les deux gonflent le residu de zone de la meme facon.
    """
    ev = evidence_frame(X, signals, thresholds)
    verdict = np.full(len(X), VERDICT_NONE, dtype=object)

    house_only = (ev["house_internal"] | ev["house_vs_peers"]).to_numpy()
    zone_bad = ev["zone_mismatch"].to_numpy()
    tz_bad = ev["trans_vs_zones_mismatch"].to_numpy()
    th_bad = ev["trans_vs_houses_mismatch"].to_numpy()

    deviates = signals["house_deviates"].to_numpy(dtype=bool)
    n_deviating = (pd.Series(deviates.astype(int))
                   .groupby(zone_key).transform("sum").to_numpy())
    n_internal = (pd.Series(ev["house_internal"].to_numpy().astype(int))
                  .groupby(zone_key).transform("sum").to_numpy())
    zone_quiet = (n_deviating == 0) & (n_internal == 0)

    # CASE 3 : maisons et transformateur d'accord, compteur de zone a l'ecart,
    # et AUCUNE maison de la zone ne se distingue -> c'est bien la zone
    verdict[zone_bad & ~th_bad & zone_quiet] = VERDICT_ZONE_METER

    # CASE 4 : maisons et zones d'accord entre elles, poste a l'ecart des deux
    verdict[~zone_bad & tz_bad & th_bad] = VERDICT_TRANS_METER

    # CASE 2 : mouvement coherent de plusieurs maisons, confirme par la zone
    verdict[(n_deviating >= thresholds.min_coherent_neighbours)
            & ~zone_bad] = VERDICT_GRID

    # CASE 1 : incoherence strictement locale a une maison
    case1 = ((house_only | deviates) & ~th_bad
             & (n_deviating <= 1)
             & ~(zone_bad & zone_quiet))
    verdict[case1] = VERDICT_LOCAL

    # CASE 5 : les niveaux se contredisent
    verdict[zone_bad & th_bad & house_only & (n_deviating >= 1)] = \
        VERDICT_AMBIGUOUS

    out = ev.copy()
    out.insert(0, "rule_verdict", verdict)
    out.insert(1, "rule_anomaly", np.isin(verdict, ANOMALY_VERDICTS))
    out["n_deviating_in_zone"] = n_deviating
    return out


def zone_level_flags(rules: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Verdict au niveau ZONE : le compteur de zone est-il en cause ?

    Les defauts de compteur de zone ne sont pas une classe apprise par LightGBM
    (les maisons ne sont pas labellisees quand seul le compteur de zone ment) :
    ils sont traites ici, par redondance entre niveaux.
    """
    frame = pd.DataFrame({
        "timestamp": table["timestamp"].to_numpy(),
        "zone_id": table["zone_id"].to_numpy(),
        "flag": (rules["rule_verdict"] == VERDICT_ZONE_METER).to_numpy(),
    })
    grouped = frame.groupby(["timestamp", "zone_id"], as_index=False)["flag"].mean()
    # la zone est declaree fautive si la majorite de ses maisons le constate
    grouped["zone_meter_fault"] = grouped["flag"] >= 0.5
    return grouped.drop(columns="flag")


def transformer_level_flags(rules: pd.DataFrame,
                            table: pd.DataFrame) -> pd.DataFrame:
    """Verdict au niveau TRANSFORMATEUR : zones coherentes, poste en ecart."""
    frame = pd.DataFrame({
        "timestamp": table["timestamp"].to_numpy(),
        "flag": (rules["rule_verdict"] == VERDICT_TRANS_METER).to_numpy(),
    })
    grouped = frame.groupby("timestamp", as_index=False)["flag"].mean()
    grouped["transformer_meter_fault"] = grouped["flag"] >= 0.5
    return grouped.drop(columns="flag")
