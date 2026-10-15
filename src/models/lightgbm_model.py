"""LightGBM multiclasse pour les classes d'anomalie apprises au niveau maison.

Le meme objet est utilise par l'approche A et par l'approche B : c'est ce qui
garantit que la seule difference entre les deux est l'apport d'Isolation Forest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


@dataclass
class TrainedLightGBM:
    model: LGBMClassifier
    classes: list[str]
    train_time_s: float
    n_train_rows: int
    features: list[str]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilites dans l'ordre canonique de ``self.classes``.

        Si une classe est absente de la periode d'entrainement, LightGBM ne
        produit pas de colonne pour elle. Le reste du systeme (fusion,
        calibration, metriques) suppose un vecteur de largeur fixe dans un
        ordre connu : on reconstruit donc la matrice complete, les classes
        jamais vues recevant une probabilite nulle.
        """
        raw = np.asarray(self.model.predict_proba(X[self.features]),
                         dtype=float)
        model_classes = list(self.model.classes_)
        if model_classes == list(self.classes):
            return raw

        full = np.zeros((len(X), len(self.classes)), dtype=float)
        index = {label: position
                 for position, label in enumerate(self.classes)}
        for column, label in enumerate(model_classes):
            if label not in index:
                raise ValueError(
                    f"classe {label!r} predite par le modele mais absente de "
                    f"l'ordre canonique {list(self.classes)}")
            full[:, index[label]] = raw[:, column]
        return full

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.asarray(self.classes)[proba.argmax(axis=1)]


#: classes difficiles identifiees par le diagnostic d'erreurs
WEAK_CLASSES = ("sensor_drift", "sensor_bias", "falsified_reading")


def balanced_sample_weights(y: np.ndarray, mode: str = "balanced") -> np.ndarray:
    """Ponderation des classes.

    ``balanced``      inversement proportionnel a l'effectif (reference)
    ``sqrt_balanced`` racine du precedent : correction plus douce
    ``boost_weak``    equilibre, puis x2 sur les trois classes difficiles
    """
    classes, counts = np.unique(y, return_counts=True)
    base = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    if mode == "sqrt_balanced":
        base = {c: float(np.sqrt(w)) for c, w in base.items()}
    elif mode == "boost_weak":
        base = {c: w * (2.0 if c in WEAK_CLASSES else 1.0)
                for c, w in base.items()}
    elif mode != "balanced":
        raise ValueError(f"mode de ponderation inconnu : {mode}")
    return np.array([base[v] for v in y], dtype=float)


def train_lightgbm(X: pd.DataFrame, y: np.ndarray, features: list[str],
                   cfg, weight_mode: str = "balanced") -> TrainedLightGBM:
    """Entraine le classifieur multiclasse sur les lignes fournies."""
    params = dict(cfg.section("model.lightgbm"))
    params["random_state"] = cfg.random_state
    classes = sorted(pd.unique(y).tolist())
    assert len(classes) >= 2, f"une seule classe dans le train : {classes}"

    model = LGBMClassifier(objective="multiclass", num_class=len(classes), **params)
    start = time.perf_counter()
    model.fit(X[features], y,
              sample_weight=balanced_sample_weights(y, weight_mode))
    elapsed = time.perf_counter() - start

    # LightGBM trie les classes ; on recupere son ordre officiel
    return TrainedLightGBM(model=model, classes=list(model.classes_),
                           train_time_s=elapsed, n_train_rows=len(X),
                           features=list(features))
