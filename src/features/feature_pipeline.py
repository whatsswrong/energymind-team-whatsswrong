"""Construction des 24 features finales d'EnergyMind.

    12 features d'origine        base_features.py
  +  4 ecarts a l'historique     baseline_features.py
  +  8 features de persistance   persistence_features.py
  = 24 features finales

Deux garanties structurelles :

1. **Aucune fuite temporelle.** Toutes les fenetres sont causales : elles
   n'utilisent que le pas courant et les pas passes. Les matrices sont
   indexees par entite, donc aucune valeur ne traverse une frontiere de maison.
2. **Aucune fuite train/test.** Toute statistique apprise (baselines, echelles
   robustes, seuils, modele de perte) est estimee sur les lignes
   d'ENTRAINEMENT seules via ``fit()``, puis appliquee telle quelle a la
   validation et au test. Le constructeur est re-ajuste pour chaque decoupage.

Les NaN sont conserves : ils portent de l'information (classe ``missing_data``)
et LightGBM les gere nativement. Seule Isolation Forest, qui ne les accepte
pas, recoit une copie imputee par les medianes du train.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.merge import WideData
from . import base_features, baseline_features, persistence_features
from .windows import fit_loss_model, robust_scale

#: liste faisant autorite : l'ordre des colonnes de la matrice X
FINAL_FEATURES: list[str] = (base_features.BASE_FEATURE_NAMES
                             + baseline_features.BASELINE_FEATURE_NAMES
                             + persistence_features.PERSISTENCE_FEATURE_NAMES)

N_FINAL_FEATURES = len(FINAL_FEATURES)

#: colonnes de redondance reservees au moteur de regles — jamais dans X
RULE_SIGNAL_NAMES = ("zone_rel", "zone_v_rel", "trans_vs_zones_rel",
                     "trans_vs_houses_rel", "house_deviates")


@dataclass
class FittedStats:
    """Statistiques apprises sur le TRAIN et appliquees partout."""

    zone_resid: tuple[float, float] = (0.0, 1.0)
    trans_resid: tuple[float, float] = (0.0, 1.0)
    # None tant que fit() n'a pas eu lieu. Surtout pas (0.0, 0.0) : c'est un
    # resultat d'ajustement parfaitement legitime, qui serait alors confondu
    # avec "pas encore ajuste" et declencherait un reajustement silencieux.
    zone_loss_model: tuple[float, float] | None = None
    trans_loss_model: tuple[float, float] | None = None
    trans_house_loss_model: tuple[float, float] | None = None
    baseline: np.ndarray | None = None      # (creneau, week-end, maison)
    house_median_P: np.ndarray | None = None
    hour_median: np.ndarray | None = None   # (24, maison)
    norm_center: np.ndarray | None = None   # (maison,)
    norm_scale: np.ndarray | None = None
    frozen_tol_P: float = 0.002
    frozen_tol_V: float = 0.002
    neighbour_z_threshold: float = 3.0
    imputation: dict = field(default_factory=dict)


class FeatureBuilder:
    """Calcule les 24 features ; ajuste ses statistiques sur le train seul."""

    def __init__(self, wide: WideData, cfg) -> None:
        self.wide = wide
        self.cfg = cfg
        self.f = cfg.section("features")
        self.nominal_v = float(cfg.get("grid.nominal_voltage_v"))
        self.capacity = float(cfg.get("grid.transformer_capacity_kw"))
        self.dt = cfg.step_hours
        self.stats = FittedStats()
        self.raw_signals = self._build_raw_signals()

    # ------------------------------------------------------------------
    # grandeurs independantes de tout apprentissage, calculees une fois
    # ------------------------------------------------------------------
    def _build_raw_signals(self) -> dict[str, np.ndarray]:
        wide = self.wide
        eps = float(self.f["eps_power_kw"])

        sum_house_P = np.zeros((wide.T, wide.Z))
        for zone in range(wide.Z):
            columns = np.flatnonzero(wide.zone_of_house == zone)
            sum_house_P[:, zone] = np.nansum(wide.P[:, columns], axis=1)
        sum_zone_P = wide.zone_P.sum(axis=1)

        return {
            "sum_house_P": sum_house_P,
            "sum_zone_P": sum_zone_P,
            "zone_rel": wide.zone_P / np.maximum(sum_house_P, eps) - 1.0,
            "trans_rel": wide.trans_P / np.maximum(sum_zone_P, eps) - 1.0,
            "dP": _causal_diff(wide.P),
            "dV": _causal_diff(wide.V),
            "dE": _causal_diff(wide.E),
        }

    # ------------------------------------------------------------------
    # apprentissage des statistiques, sur le train uniquement
    # ------------------------------------------------------------------
    def fit(self, train_steps: np.ndarray) -> "FeatureBuilder":
        """``train_steps`` : indices des pas de temps du train."""
        self._fit_loss_models(train_steps)
        self._fit_baselines(train_steps)
        self._fit_robust_scales(train_steps)
        self._fit_thresholds(train_steps)
        return self

    def _fit_loss_models(self, train_steps: np.ndarray) -> None:
        """Perte attendue en fonction de la charge, puis echelle du residu."""
        raw = self.raw_signals
        self.stats.zone_loss_model = fit_loss_model(
            raw["sum_house_P"][train_steps].ravel(),
            raw["zone_rel"][train_steps].ravel())
        self.stats.trans_loss_model = fit_loss_model(
            raw["sum_zone_P"][train_steps], raw["trans_rel"][train_steps])

        # Redondance transformateur <-> somme des maisons, utilisee par les
        # regles. Elle doit etre apprise sur le TRAIN comme les autres : la
        # fitter sur l'historique complet ferait entrer la periode de test
        # dans un parametre utilise a l'inference.
        total_houses = raw["sum_house_P"].sum(axis=1)
        trans_house_rel = self._trans_house_relative(total_houses)
        self.stats.trans_house_loss_model = fit_loss_model(
            total_houses[train_steps], trans_house_rel[train_steps])
        self.stats.zone_resid = robust_scale(
            self.zone_residual()[train_steps].ravel())
        self.stats.trans_resid = robust_scale(
            self.trans_residual()[train_steps])

    def _fit_baselines(self, train_steps: np.ndarray) -> None:
        """Medianes de consommation par creneau et par heure, sur le train."""
        wide = self.wide
        eps = float(self.f["eps_power_kw"])
        n_slots = int(wide.slot.max()) + 1
        slot_train = wide.slot[train_steps]
        weekend_train = wide.is_weekend[train_steps].astype(int)
        power_train = wide.P[train_steps]

        baseline = np.full((n_slots, 2, wide.H), np.nan)
        counts = np.zeros((n_slots, 2, wide.H))
        for slot in range(n_slots):
            for weekend in (0, 1):
                selected = (slot_train == slot) & (weekend_train == weekend)
                if selected.any():
                    block = power_train[selected]
                    counts[slot, weekend] = np.sum(~np.isnan(block), axis=0)
                    with np.errstate(invalid="ignore"):
                        baseline[slot, weekend] = np.nanmedian(block, axis=0)
        house_median = np.nanmedian(power_train, axis=0)

        minutes_per_step = int(self.cfg.get("grid.interval_minutes"))
        hours_train = (slot_train * minutes_per_step) // 60
        hour_median = np.full((24, wide.H), np.nan)
        for hour in range(24):
            selected = hours_train == hour
            if selected.any():
                with np.errstate(invalid="ignore"):
                    hour_median[hour] = np.nanmedian(power_train[selected], axis=0)
        self.stats.hour_median = np.maximum(
            np.where(np.isfinite(hour_median), hour_median,
                     house_median[None, :]), eps)

        too_few = counts < int(self.f["baseline_min_samples"])
        baseline = np.where(too_few | ~np.isfinite(baseline),
                            house_median[None, None, :], baseline)
        self.stats.baseline = np.maximum(baseline, eps)
        self.stats.house_median_P = np.maximum(house_median, eps)

    def _fit_robust_scales(self, train_steps: np.ndarray) -> None:
        """Echelle robuste de la consommation normalisee, maison par maison."""
        normalised_train = self.normalised_power(train_steps)
        centres = np.zeros(self.wide.H)
        scales = np.ones(self.wide.H)
        for house in range(self.wide.H):
            centres[house], scales[house] = robust_scale(normalised_train[:, house])
        self.stats.norm_center, self.stats.norm_scale = centres, scales

    def _fit_thresholds(self, train_steps: np.ndarray) -> None:
        """Tolerance "compteur fige" et seuil "voisin anormal"."""
        wide, raw = self.wide, self.raw_signals
        eps = float(self.f["eps_power_kw"])
        rel_dP = np.abs(raw["dP"][train_steps]) / np.maximum(
            np.abs(wide.P[train_steps]), eps)
        rel_dV = np.abs(raw["dV"][train_steps]) / np.maximum(
            np.abs(wide.V[train_steps]), 1.0)
        quantile = float(self.f["frozen_tol_quantile"])
        floor = float(self.f["frozen_tol_floor"])
        self.stats.frozen_tol_P = max(float(np.nanquantile(rel_dP, quantile)),
                                      floor)
        self.stats.frozen_tol_V = max(float(np.nanquantile(rel_dV, quantile)),
                                      floor)

        normalised_train = self.normalised_power(train_steps)
        z_train = np.abs((normalised_train - self.stats.norm_center)
                         / np.maximum(self.stats.norm_scale, 1e-9))
        self.stats.neighbour_z_threshold = float(
            np.nanquantile(z_train, float(self.f["neighbor_z_quantile"])))

    # ------------------------------------------------------------------
    # grandeurs derivees, partagees par les modules de features
    # ------------------------------------------------------------------
    def _trans_house_relative(self, total_houses: np.ndarray) -> np.ndarray:
        """(T,) ecart relatif entre le compteur du poste et la somme des maisons."""
        eps = float(self.f["eps_power_kw"])
        return self.wide.trans_P / np.maximum(total_houses, eps) - 1.0

    def zone_residual(self) -> np.ndarray:
        """(T, Z) ecart entre la mesure de zone et la perte attendue."""
        if self.stats.zone_loss_model is None:
            raise RuntimeError("appeler fit(train_steps) avant zone_residual()")
        intercept, slope = self.stats.zone_loss_model
        return self.raw_signals["zone_rel"] - (
            intercept + slope * self.raw_signals["sum_house_P"])

    def trans_residual(self) -> np.ndarray:
        """(T,) ecart entre la mesure du poste et la perte attendue."""
        if self.stats.trans_loss_model is None:
            raise RuntimeError("appeler fit(train_steps) avant trans_residual()")
        intercept, slope = self.stats.trans_loss_model
        return self.raw_signals["trans_rel"] - (
            intercept + slope * self.raw_signals["sum_zone_P"])

    def normalised_power(self, steps=slice(None)) -> np.ndarray:
        """P divisee par la baseline apprise : maisons rendues comparables."""
        wide = self.wide
        slot = wide.slot[steps]
        weekend = wide.is_weekend[steps].astype(int)
        baseline = self.stats.baseline[slot, weekend]
        return wide.P[steps] / np.maximum(baseline,
                                          float(self.f["eps_power_kw"]))

    # ------------------------------------------------------------------
    # matrice de features
    # ------------------------------------------------------------------
    def transform(self) -> pd.DataFrame:
        """Construit les 24 features finales, dans l'ordre de FINAL_FEATURES."""
        assert self.stats.baseline is not None, "appeler fit() avant transform()"
        wide = self.wide

        columns = base_features.build(self)
        columns.update(baseline_features.build(self))
        columns.update(persistence_features.build(
            self, columns["res_zone"], columns["nb_voisins_anormaux"]))

        assert list(columns) == FINAL_FEATURES, (
            f"ordre des features incoherent : {list(columns)}")
        frame = pd.DataFrame({name: wide.flatten(values)
                              for name, values in columns.items()})
        return frame.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # signaux de redondance reserves au moteur de regles
    # ------------------------------------------------------------------
    def rule_signals(self) -> pd.DataFrame:
        """Residus relatifs entre niveaux de mesure, pour les regles.

        Ce ne sont PAS des features du modele : le moteur de regles compare
        trois agregats independants les uns des autres.

        ``trans_vs_houses`` est la cle du diagnostic : le transformateur et la
        somme des compteurs de maison sont deux mesures *independantes des
        compteurs de zone*. Si elles s'accordent alors que le compteur de zone
        s'ecarte, c'est lui qui ment ; si elles divergent toutes deux pendant
        que les zones s'accordent, c'est le compteur du poste.
        """
        wide, raw = self.wide, self.raw_signals
        eps = float(self.f["eps_power_kw"])
        zone_of_house = wide.zone_of_house

        if self.stats.trans_house_loss_model is None:
            raise RuntimeError(
                "FeatureBuilder.fit(train_steps) doit etre appele avant "
                "rule_signals() : sans cela le modele de perte "
                "transformateur <-> maisons serait ajuste sur l'historique "
                "complet, ce qui ferait fuiter la periode de test.")

        total_houses = raw["sum_house_P"].sum(axis=1)
        zone_rel = self.zone_residual()[:, zone_of_house]
        trans_vs_zones = np.repeat(self.trans_residual()[:, None], wide.H,
                                   axis=1)
        trans_house_rel = self._trans_house_relative(total_houses)
        intercept, slope = self.stats.trans_house_loss_model
        trans_vs_houses = np.repeat(
            (trans_house_rel - (intercept + slope * total_houses))[:, None],
            wide.H, axis=1)

        # redondance en TENSION : le compteur de zone face a la mediane des
        # tensions de ses maisons. Un creux reel fait baisser les deux ; un
        # compteur de zone fausse ne fait bouger que lui.
        zone_v_rel = np.full((wide.T, wide.Z), np.nan)
        for zone in range(wide.Z):
            columns = np.flatnonzero(wide.zone_of_house == zone)
            with np.errstate(invalid="ignore"):
                house_voltage = np.nanmedian(wide.V[:, columns], axis=1)
            zone_v_rel[:, zone] = wide.zone_V[:, zone] / np.maximum(
                house_voltage, 1.0) - 1.0

        normalised = self.normalised_power()
        z_normalised = (normalised - self.stats.norm_center) / np.maximum(
            self.stats.norm_scale, 1e-9)
        deviates = np.abs(z_normalised) > self.stats.neighbour_z_threshold

        return pd.DataFrame({
            "zone_rel": wide.flatten(zone_rel),
            "zone_v_rel": wide.flatten(zone_v_rel[:, zone_of_house]),
            "trans_vs_zones_rel": wide.flatten(trans_vs_zones),
            "trans_vs_houses_rel": wide.flatten(trans_vs_houses),
            "house_deviates": wide.flatten(deviates.astype(float)) > 0.5,
        }).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # imputation, pour les modeles qui n'acceptent pas les NaN
    # ------------------------------------------------------------------
    def fit_imputation(self, X: pd.DataFrame, train_rows: np.ndarray) -> None:
        """Medianes du train, utilisees uniquement par Isolation Forest."""
        self.stats.imputation = {
            column: (float(np.nanmedian(X.loc[train_rows, column]))
                     if np.isfinite(X.loc[train_rows, column]).any() else 0.0)
            for column in X.columns}

    def impute(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.stats.imputation, "appeler fit_imputation() d'abord"
        return X.fillna(value=self.stats.imputation)


def _causal_diff(values: np.ndarray) -> np.ndarray:
    """Difference avec le pas precedent ; NaN au premier pas."""
    first = np.full((1, values.shape[1]), np.nan)
    return np.vstack([first, np.diff(values, axis=0)])


def feature_report(X: pd.DataFrame) -> pd.DataFrame:
    """Controle de sante de chaque feature, avant tout entrainement."""
    rows = []
    for name in X.columns:
        column = X[name]
        finite = column[np.isfinite(column)]
        has_values = len(finite) > 0
        rows.append({
            "feature": name,
            "missing_pct": round(100.0 * column.isna().mean(), 3),
            "infinite": int(np.isinf(column.to_numpy(dtype=float)).sum()),
            "min": round(float(finite.min()), 4) if has_values else np.nan,
            "p01": round(float(finite.quantile(0.01)), 4) if has_values else np.nan,
            "median": round(float(finite.median()), 4) if has_values else np.nan,
            "mean": round(float(finite.mean()), 4) if has_values else np.nan,
            "p99": round(float(finite.quantile(0.99)), 4) if has_values else np.nan,
            "max": round(float(finite.max()), 4) if has_values else np.nan,
            "std": round(float(finite.std()), 4) if has_values else np.nan,
        })
    return pd.DataFrame(rows)
