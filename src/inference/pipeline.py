"""Pipeline d'inference EnergyMind : des mesures brutes au verdict.

    mesures brutes
      -> assemblage maison + zone + transformateur
      -> 24 features
      -> LightGBM            probabilites par classe connue
      -> Vector Scaling      confiance calibree
                                                  }
      -> Isolation Forest    score de nouveaute   }  fusion -> verdict
      -> regles physiques    preuve de coherence  }

Repartition des roles, volontairement etanche :

* **LightGBM** classe les anomalies CONNUES. Lui seul produit un nom de classe.
* **Isolation Forest** ne produit qu'un score d'atypicite. Il ne classe rien et
  n'invente jamais de nom de classe.
* **Les regles physiques** comparent les niveaux de mesure entre eux et
  produisent une preuve de coherence ; elles n'apprennent rien.
* **La calibration** ne change que l'expression de la confiance.

Politique de confiance : au-dessous du seuil configure (0,80), le systeme
repond INCERTAIN plutot qu'un diagnostic auquel il ne croit pas lui-meme.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ..data.merge import WideData
from ..data.splits import Split, assert_no_leakage
from ..features.feature_pipeline import (FINAL_FEATURES, FeatureBuilder,
                                         feature_report)
from ..models.isolation_forest import (TrainedIsolationForest,
                                       train_isolation_forest)
from ..models.lightgbm_model import TrainedLightGBM, train_lightgbm
from ..rules.physical_rules import apply_rules, fit_thresholds

NORMAL = "normal"
UNCERTAIN = "INCERTAIN"
#: sortie lorsque les preuves disent "anomalie" mais que LightGBM ne reconnait
#: aucune classe : on ne devine jamais un nom de classe
UNKNOWN_ANOMALY = "unknown_anomaly"


# ---------------------------------------------------------------------------
# preparation des donnees d'un decoupage
# ---------------------------------------------------------------------------

@dataclass
class PreparedSplit:
    """Tout ce dont le modele a besoin pour un decoupage temporel donne."""

    name: str
    table: pd.DataFrame
    X: pd.DataFrame                 # 24 features, NaN conserves
    X_imputed: pd.DataFrame         # version imputee (medianes du train)
    y: np.ndarray
    rules: pd.DataFrame
    signals: pd.DataFrame
    thresholds: object
    features: list[str]
    rows: dict[str, np.ndarray]


def calibration_mask(table: pd.DataFrame, raw) -> tuple[np.ndarray, np.ndarray]:
    """Lignes utilisables pour calibrer les seuils des regles.

    Il ne suffit pas que la maison soit normale : si un VOISIN est en coupure,
    la somme de la zone chute et le residu de zone de cette maison "normale"
    devient enorme. Calibrer dessus ferait exploser les seuils. On exige donc
    un instant propre a tous les niveaux.

    Retourne ``(propre_au_niveau_zone, propre_sur_tout_le_reseau)``.
    """
    house_normal = (table["label"] == "normal").to_numpy()

    zone_all_normal = (table.assign(_normal=house_normal)
                       .groupby(["timestamp", "zone_id"])["_normal"]
                       .transform("all").to_numpy())

    zone_ok = raw.zone_labels.assign(_ok=lambda d: d["label"] == "normal")
    zone_map = table.merge(zone_ok[["timestamp", "zone_id", "_ok"]],
                           on=["timestamp", "zone_id"], how="left",
                           validate="many_to_one")["_ok"].fillna(False).to_numpy()

    trans_ok = raw.transformer_labels.assign(_ok=lambda d: d["label"] == "normal")
    trans_map = table.merge(trans_ok[["timestamp", "_ok"]], on="timestamp",
                            how="left", validate="many_to_one"
                            )["_ok"].fillna(False).to_numpy()

    zone_clean = house_normal & zone_all_normal & zone_map & trans_map
    global_clean = (pd.Series(zone_clean, index=table.index)
                    .groupby(table["timestamp"].to_numpy()).transform("all")
                    .to_numpy())
    return zone_clean, global_clean


def prepare_split(cfg, table: pd.DataFrame, wide: WideData, split: Split,
                  raw) -> tuple[PreparedSplit, FeatureBuilder]:
    """Construit features et regles d'un decoupage, sans aucune fuite.

    Les statistiques (baselines, echelles, seuils) sont ajustees sur le train
    de CE decoupage uniquement, puis appliquees partout.
    """
    assert_no_leakage(split, table)

    builder = FeatureBuilder(wide, cfg)
    builder.fit(split.steps["train"])
    X = builder.transform()
    assert len(X) == len(table), "features et table desalignees"
    # l'ordre des lignes est contractuel : on le verifie sur une mesure brute
    assert np.allclose(wide.flatten(wide.P), table["P"].to_numpy(),
                       equal_nan=True), "ordre des lignes incoherent"

    builder.fit_imputation(X, split.rows["train"])
    X_imputed = builder.impute(X)
    assert not X_imputed.isna().any().any(), "imputation incomplete"

    signals = builder.rule_signals()
    zone_clean, global_clean = calibration_mask(table, raw)
    thresholds = fit_thresholds(X, signals, zone_clean, global_clean,
                                split.rows["train"], cfg)
    zone_key = (table["timestamp"].astype("int64").to_numpy() * 10
                + pd.factorize(table["zone_id"])[0])
    rules = apply_rules(X, signals, thresholds, zone_key)

    prepared = PreparedSplit(
        name=split.name, table=table, X=X, X_imputed=X_imputed,
        y=table["label"].to_numpy(), rules=rules, signals=signals,
        thresholds=thresholds, features=list(FINAL_FEATURES),
        rows=split.rows)
    return prepared, builder


# ---------------------------------------------------------------------------
# modele complet
# ---------------------------------------------------------------------------

@dataclass
class FusionThresholds:
    """Seuils de fusion, choisis sur la validation uniquement."""

    promote_min_prob: float
    demote_max_prob: float
    validation_macro_f1: float


@dataclass
class EnergyMindModel:
    """LightGBM + Isolation Forest + regles physiques, avec leur fusion."""

    cfg: object
    lightgbm: TrainedLightGBM | None = None
    isolation_forest: TrainedIsolationForest | None = None
    fusion: FusionThresholds | None = None
    emit_unknown: bool = False
    fit_time_s: float = 0.0
    inference_time_s: float = field(default=0.0)

    # ---- entrainement -------------------------------------------------
    def fit(self, prepared: PreparedSplit) -> "EnergyMindModel":
        """Entraine les deux modeles, puis regle la fusion sur la validation."""
        start = time.perf_counter()
        train_rows = prepared.rows["train"]

        self.lightgbm = train_lightgbm(
            prepared.X.loc[train_rows], prepared.y[train_rows],
            prepared.features, self.cfg)
        self.isolation_forest = train_isolation_forest(
            prepared.X_imputed.loc[train_rows], prepared.y[train_rows],
            prepared.features, self.cfg)

        self.fusion = self._tune_fusion(prepared, prepared.rows["validation"])
        self.fit_time_s = time.perf_counter() - start
        return self

    def _tune_fusion(self, prepared: PreparedSplit,
                     validation_rows: np.ndarray) -> FusionThresholds:
        """Recherche exhaustive des deux seuils, sur la validation seulement."""
        signals = self.component_outputs(prepared, validation_rows)
        y_validation = prepared.y[validation_rows]
        best = FusionThresholds(0.15, 0.60, -1.0)
        for promote in self.cfg.get("fusion.promote_min_prob_grid"):
            for demote in self.cfg.get("fusion.demote_max_prob_grid"):
                prediction = fuse(signals, float(promote), float(demote))
                score = f1_score(y_validation, prediction, average="macro",
                                 zero_division=0)
                if score > best.validation_macro_f1:
                    best = FusionThresholds(float(promote), float(demote),
                                            float(score))
        return best

    # ---- sorties des composants ---------------------------------------
    def component_outputs(self, prepared: PreparedSplit,
                          rows: np.ndarray) -> dict:
        """Sorties brutes de chaque composant, avant fusion."""
        X = prepared.X.loc[rows]
        proba = self.lightgbm.predict_proba(X)
        classes = np.asarray(self.lightgbm.classes)

        normal_index = (int(np.flatnonzero(classes == NORMAL)[0])
                        if NORMAL in classes else -1)
        anomaly_columns = np.flatnonzero(classes != NORMAL)
        best_anomaly = anomaly_columns[proba[:, anomaly_columns].argmax(axis=1)]

        rule_anomaly = prepared.rules.loc[rows, "rule_anomaly"].to_numpy(bool)
        if_score = self.isolation_forest.score(prepared.X_imputed.loc[rows])
        if_anomalous = if_score < self.isolation_forest.threshold

        return {
            "classes": classes,
            "proba": proba,
            "lgbm_pred": classes[proba.argmax(axis=1)],
            "p_max": proba.max(axis=1),
            "p_normal": (proba[:, normal_index] if normal_index >= 0
                         else np.zeros(len(proba))),
            "best_anom_class": classes[best_anomaly],
            "p_best_anom": proba[np.arange(len(proba)), best_anomaly],
            "rule_anomaly": rule_anomaly,
            "if_score": if_score,
            "if_anomalous": if_anomalous,
            "evidence": rule_anomaly | if_anomalous,
        }

    # ---- prediction ----------------------------------------------------
    def predict(self, prepared: PreparedSplit, rows: np.ndarray) -> pd.DataFrame:
        """Verdict final, accompagne de toutes les pieces du raisonnement."""
        assert self.fusion is not None, "appeler fit() avant predict()"
        start = time.perf_counter()
        signals = self.component_outputs(prepared, rows)
        prediction = fuse(signals, self.fusion.promote_min_prob,
                          self.fusion.demote_max_prob, self.emit_unknown)
        self.inference_time_s = time.perf_counter() - start

        table = prepared.table.loc[rows]
        return pd.DataFrame({
            "timestamp": table["timestamp"].to_numpy(),
            "house_id": table["house_id"].to_numpy(),
            "zone_id": table["zone_id"].to_numpy(),
            "event_id": table["event_id"].to_numpy(),
            "y_true": prepared.y[rows],
            "y_pred": prediction,
            "lgbm_pred": signals["lgbm_pred"],
            "p_max": signals["p_max"],
            "p_normal": signals["p_normal"],
            "rule_verdict": prepared.rules.loc[rows, "rule_verdict"].to_numpy(),
            "rule_anomaly": signals["rule_anomaly"],
            "if_score": signals["if_score"],
            "if_anomalous": signals["if_anomalous"],
            "evidence": signals["evidence"],
        })

    @property
    def n_components(self) -> int:
        return 3

    @property
    def description(self) -> str:
        return "LightGBM + Isolation Forest + regles physiques"


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------

def fuse(signals: dict, promote_min_prob: float, demote_max_prob: float,
         emit_unknown: bool = False) -> np.ndarray:
    """Combine modele et preuves en un verdict unique.

    * LightGBM dit "normal" MAIS les preuves signalent une anomalie et sa
      meilleure classe anormale atteint ``promote_min_prob`` -> on promeut.
    * LightGBM dit "anormal" SANS preuve et avec une confiance inferieure a
      ``demote_max_prob`` -> on retombe sur "normal".

    ``emit_unknown`` respecte la regle de conception : lors d'une promotion, le
    systeme repond UNKNOWN plutot que d'inventer un nom de classe. Mesure sur
    le modele final : la classe devinee n'etait correcte que dans 17,8 % des
    cas, sur 45 lignes concernees.
    """
    # dtype objet : un tableau numpy de largeur fixe tronquerait silencieusement
    # une etiquette plus longue que les noms de classes
    prediction = signals["lgbm_pred"].astype(object)
    says_normal = prediction == NORMAL
    evidence = signals["evidence"]

    promote = (says_normal & evidence
               & (signals["p_best_anom"] >= promote_min_prob))
    prediction[promote] = (UNKNOWN_ANOMALY if emit_unknown
                           else signals["best_anom_class"][promote])

    demote = (~says_normal & ~evidence & (signals["p_max"] < demote_max_prob))
    prediction[demote] = NORMAL
    return prediction


# ---------------------------------------------------------------------------
# politique de confiance (face exploitant)
# ---------------------------------------------------------------------------

def apply_confidence_policy(predictions: pd.DataFrame,
                            calibrated_confidence: np.ndarray,
                            threshold: float) -> pd.DataFrame:
    """Ajoute le statut operationnel a partir de la confiance calibree.

    Au-dessous du seuil, le diagnostic de classe n'est pas affiche : le systeme
    repond INCERTAIN. L'information "anomalie ou non" est conservee, seul le
    nom de la classe est retenu.
    """
    out = predictions.copy()
    out["confidence"] = calibrated_confidence
    confident = calibrated_confidence >= threshold
    out["status"] = np.where(confident, "DIAGNOSTIC", UNCERTAIN)
    out["reported_label"] = np.where(confident, out["y_pred"], UNCERTAIN)
    return out


__all__ = ["PreparedSplit", "EnergyMindModel", "FusionThresholds",
           "prepare_split", "calibration_mask", "fuse",
           "apply_confidence_policy", "feature_report",
           "NORMAL", "UNCERTAIN", "UNKNOWN_ANOMALY", "FINAL_FEATURES"]
