"""Isolation Forest : modele du comportement NORMAL, pas un classifieur.

Il est entraine uniquement sur les lignes normales du TRAIN, donc il ne connait
aucune classe d'anomalie. C'est precisement ce qui le rend interessant pour
l'experience "anomalie inconnue" : son signal ne depend pas du fait que
LightGBM ait deja vu la classe.

Le seuil de decision est le quantile des scores d'entrainement correspondant au
taux de faux positifs vise — estime sur le TRAIN seul, jamais sur le test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class TrainedIsolationForest:
    model: IsolationForest
    threshold: float          # score en dessous duquel la ligne est "anormale"
    train_time_s: float
    n_train_rows: int
    features: list[str]
    target_train_fpr: float

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Score de normalite : plus il est bas, plus la ligne est atypique."""
        return self.model.score_samples(X[self.features])

    def is_anomalous(self, X: pd.DataFrame) -> np.ndarray:
        return self.score(X) < self.threshold


def train_isolation_forest(X_imputed: pd.DataFrame, y: np.ndarray,
                           features: list[str], cfg) -> TrainedIsolationForest:
    """Entraine sur les lignes NORMALES uniquement et fixe le seuil sur le train."""
    params = dict(cfg.section("model.isolation_forest"))
    target_fpr = float(params.pop("target_train_fpr"))
    params["random_state"] = cfg.random_state

    normal_mask = (y == "normal")
    assert normal_mask.sum() > 1000, "trop peu de lignes normales pour l'IF"
    X_normal = X_imputed.loc[normal_mask, features]

    model = IsolationForest(**params)
    start = time.perf_counter()
    model.fit(X_normal)
    elapsed = time.perf_counter() - start

    # seuil : le quantile des scores sur le normal d'entrainement qui laisse
    # passer exactement la fraction de faux positifs visee
    scores_normal = model.score_samples(X_normal)
    threshold = float(np.quantile(scores_normal, target_fpr))

    return TrainedIsolationForest(model=model, threshold=threshold,
                                  train_time_s=elapsed,
                                  n_train_rows=int(normal_mask.sum()),
                                  features=list(features),
                                  target_train_fpr=target_fpr)
