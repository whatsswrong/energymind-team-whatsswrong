"""Primitives de fenetre glissante, toutes causales.

Une fenetre causale n'utilise que le pas courant et les pas passes. Ces
fonctions sont le seul endroit du projet ou une fenetre temporelle est
calculee : centraliser ici garantit qu'aucune variante n'introduira de fuite
vers le futur.

Les matrices sont de forme ``(T, H)`` : une ligne par pas de temps, une colonne
par maison. Les operations sont donc appliquees independamment a chaque maison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["rolling_std", "rolling_mean", "rolling_median", "rolling_slope",
           "run_length", "neighbour_median", "robust_scale", "fit_loss_model"]


def rolling_std(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Ecart-type glissant causal, colonne par colonne."""
    return (pd.DataFrame(values)
            .rolling(window=window, min_periods=min_periods).std().to_numpy())


def rolling_mean(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Moyenne glissante causale, colonne par colonne."""
    return (pd.DataFrame(values)
            .rolling(window=window, min_periods=min_periods).mean().to_numpy())


def rolling_median(values: np.ndarray, window: int,
                   min_periods: int) -> np.ndarray:
    """Mediane glissante causale, colonne par colonne."""
    return (pd.DataFrame(values)
            .rolling(window=window, min_periods=min_periods).median().to_numpy())


def rolling_slope(values: np.ndarray, window: int,
                  min_periods: int) -> np.ndarray:
    """Pente OLS glissante causale, colonne par colonne.

    La pente d'une regression sur ``x = 0..w-1`` est une combinaison lineaire
    fixe des points de la fenetre : on l'obtient par convolution, en O(T) au
    lieu d'un ``rolling.apply`` fenetre par fenetre.
    """
    x = np.arange(window, dtype=float)
    centred = x - x.mean()
    kernel = centred / (centred ** 2).sum()

    # Les trous sont propages vers l'avant pour que la convolution reste
    # definie ; le noyau est de somme nulle, un decalage constant n'a donc
    # aucun effet sur la pente.
    #
    # ATTENTION : la validite se compte sur les valeurs BRUTES, jamais sur les
    # valeurs remplies. Pendant une coupure de capteur, le report avant cree
    # des valeurs constantes qui produiraient une pente nulle credible alors
    # qu'aucune mesure n'a ete recue. On exige donc min_periods observations
    # REELLES dans la fenetre.
    raw = np.asarray(values, dtype=float)
    filled = pd.DataFrame(raw).ffill().to_numpy()
    out = np.full_like(filled, np.nan, dtype=float)
    valid = (~np.isnan(raw)).astype(float)
    counts = pd.DataFrame(valid).rolling(window, min_periods=1).sum().to_numpy()
    filled = np.nan_to_num(filled, nan=0.0)
    for column in range(filled.shape[1]):
        convolved = np.convolve(filled[:, column], kernel[::-1],
                                mode="full")[:filled.shape[0]]
        out[:, column] = convolved
    out[counts < min_periods] = np.nan
    out[:window - 1] = np.nan
    return out


def run_length(flag: np.ndarray) -> np.ndarray:
    """Longueur du palier courant de ``True``, remise a zero sur ``False``."""
    out = np.zeros(flag.shape, dtype=float)
    for step in range(1, flag.shape[0]):
        out[step] = np.where(flag[step], out[step - 1] + 1.0, 0.0)
    return out


def neighbour_median(values: np.ndarray, wide) -> np.ndarray:
    """Mediane des AUTRES maisons de la zone, pour chaque maison."""
    out = np.full_like(values, np.nan, dtype=float)
    for house in range(wide.H):
        peers = wide.neighbours(house)
        if peers.size:
            with np.errstate(invalid="ignore"):
                out[:, house] = np.nanmedian(values[:, peers], axis=1)
    return out


def robust_scale(values: np.ndarray) -> tuple[float, float]:
    """Centre et echelle robustes (mediane, MAD), insensibles aux anomalies."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        std = float(np.std(finite))
        scale = std if std > 1e-9 else 1.0
    return median, scale


def fit_loss_model(load: np.ndarray, relative: np.ndarray) -> tuple[float, float]:
    """Perte technique attendue, affine en charge : ``rel ~ a + b x charge``.

    Les pertes Joule croissent avec la charge : un residu centre sur une
    constante exploserait en pointe et ferait passer toute surcharge pour une
    incoherence de comptage. L'ajustement se fait sur le coeur de la
    distribution (deciles 10-90) pour ne pas etre tire par les anomalies.

    Robustesse : longueurs incompatibles rejetees, valeurs non finies
    ecartees, et repli sur une pente nulle quand la regression n'a pas de
    sens — moins de deux points exploitables, ou charge quasi constante
    (``np.polyfit`` y est numeriquement instable et emet un RankWarning).
    """
    load_array = np.asarray(load, dtype=float).ravel()
    relative_array = np.asarray(relative, dtype=float).ravel()
    if load_array.size != relative_array.size:
        raise ValueError(
            f"fit_loss_model : longueurs incompatibles "
            f"({load_array.size} charges, {relative_array.size} residus)")

    usable = np.isfinite(load_array) & np.isfinite(relative_array)
    if usable.sum() < 2:
        return _fallback_intercept(relative_array[usable]), 0.0

    low, high = np.nanquantile(relative_array[usable], [0.10, 0.90])
    core = usable & (relative_array >= low) & (relative_array <= high)
    if core.sum() < 50:
        core = usable

    # une charge constante ne porte aucune information de pente
    if core.sum() < 2 or float(np.ptp(load_array[core])) < 1e-6:
        return _fallback_intercept(relative_array[core]), 0.0

    try:
        slope, intercept = np.polyfit(load_array[core], relative_array[core], 1)
    except (np.linalg.LinAlgError, ValueError):
        return _fallback_intercept(relative_array[core]), 0.0
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return _fallback_intercept(relative_array[core]), 0.0
    return float(intercept), float(slope)


def _fallback_intercept(values: np.ndarray) -> float:
    """Repli : residu median, ou zero si rien d'exploitable."""
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return median if np.isfinite(median) else 0.0
