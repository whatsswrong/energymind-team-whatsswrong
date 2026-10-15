"""Calibration de confiance des probabilites multiclasses.

Un modele peut etre exact et mal calibre : annoncer 0,90 sur des lignes qui ne
sont correctes que 60 % du temps. Pour EnergyMind, c'est le point qui decide si
un exploitant peut faire confiance a une alerte ou doit la verifier.

Trois methodes, de la plus contrainte a la plus souple :

``temperature``  un seul parametre T : ``softmax(log p / T)``. Ne change aucun
                 classement, donc n'altere ni l'exactitude ni le F1.
``vector``       un facteur et un biais par classe. Peut corriger des classes
                 systematiquement trop confiantes, au prix de 2C parametres.
``isotonic``     regression isotone un-contre-tous, non parametrique. La plus
                 souple, donc la plus exposee au surapprentissage de la
                 validation.

Toutes sont ajustees sur la VALIDATION et jamais sur le test. La selection se
fait par log-vraisemblance (regle de score propre), l'evaluation par ECE,
Brier et diagrammes de fiabilite sur le test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

EPS = 1e-12


def _safe_log(p: np.ndarray) -> np.ndarray:
    """Logarithme borne des deux cotes.

    Une probabilite exactement nulle donnerait ``-inf``, et exactement 1 un
    logit nul face a des ``-inf`` : les deux font echouer l'optimisation. On
    borne donc dans ``[EPS, 1 - EPS]``.
    """
    return np.log(np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS))


def _softmax(z: np.ndarray) -> np.ndarray:
    """Softmax stable : le maximum est retire avant l'exponentielle."""
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), EPS)


# ===== BUG 09 : assainissement des probabilites en entree ================

def sanitize_proba(proba: np.ndarray, n_classes: int | None = None
                   ) -> np.ndarray:
    """Rend exploitable une matrice de probabilites potentiellement invalide.

    Un modele amont peut produire des NaN, des infinis, des valeurs negatives
    ou des lignes non normalisees. Plutot que de propager l'anomalie jusqu'aux
    metriques, on la corrige explicitement :

    1. conversion en flottants ;
    2. NaN et infinis remplaces (``+inf`` -> 1, ``-inf`` et NaN -> 0) ;
    3. valeurs negatives ramenees a zero ;
    4. ligne entierement nulle remplacee par la distribution uniforme ;
    5. chaque ligne renormalisee a somme 1.

    Une matrice dont le nombre de colonnes ne correspond pas au nombre de
    classes attendu est une erreur de programmation, pas une donnee a
    reparer : elle leve ``ValueError``.
    """
    array = np.asarray(proba, dtype=float)
    if array.ndim != 2:
        raise ValueError(
            f"probabilites : matrice 2D attendue, forme recue {array.shape}")
    if n_classes is not None and array.shape[1] != n_classes:
        raise ValueError(
            f"probabilites : {array.shape[1]} colonnes recues, "
            f"{n_classes} classes attendues")

    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    array = np.clip(array, 0.0, None)

    totals = array.sum(axis=1, keepdims=True)
    empty = totals[:, 0] <= EPS
    if empty.any():
        array[empty] = 1.0 / array.shape[1]
        totals = array.sum(axis=1, keepdims=True)
    return array / np.maximum(totals, EPS)


def _one_hot(y: np.ndarray, classes) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}
    out = np.zeros((len(y), len(classes)))
    out[np.arange(len(y)), [index[v] for v in y]] = 1.0
    return out


# ---------------------------------------------------------------------------
# calibrateurs
# ---------------------------------------------------------------------------

class Calibrator:
    name = "base"

    def fit(self, proba: np.ndarray, y: np.ndarray, classes) -> "Calibrator":
        raise NotImplementedError

    def transform(self, proba: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def n_parameters(self) -> int:
        return 0

    def describe(self) -> str:
        return self.name


class Identity(Calibrator):
    name = "aucune"

    def fit(self, proba, y, classes):
        return self

    def transform(self, proba):
        return sanitize_proba(proba)


class TemperatureScaling(Calibrator):
    """Un seul parametre : aucune prediction ne change de classe."""

    name = "temperature"

    def __init__(self):
        self.temperature = 1.0

    def fit(self, proba, y, classes):
        logits = _safe_log(sanitize_proba(proba, len(classes)))
        target = _one_hot(y, classes)

        def nll(log_t):
            scaled = _softmax(logits / np.exp(log_t[0]))
            return -float((target * _safe_log(scaled)).sum() / len(y))

        result = minimize(nll, x0=np.array([0.0]), method="Nelder-Mead",
                          options={"xatol": 1e-4, "fatol": 1e-6})
        self.temperature = float(np.exp(result.x[0]))
        return self

    def transform(self, proba):
        return _softmax(_safe_log(sanitize_proba(proba)) / self.temperature)

    @property
    def n_parameters(self):
        return 1

    def describe(self):
        return f"temperature T={self.temperature:.3f}"


class VectorScaling(Calibrator):
    """Un facteur et un biais par classe : corrige les exces par classe."""

    name = "vector"

    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, proba, y, classes):
        logits = _safe_log(sanitize_proba(proba, len(classes)))
        target = _one_hot(y, classes)
        n_classes = len(classes)

        def nll(params):
            w, b = params[:n_classes], params[n_classes:]
            scaled = _softmax(logits * w + b)
            return -float((target * _safe_log(scaled)).sum() / len(y))

        x0 = np.concatenate([np.ones(n_classes), np.zeros(n_classes)])
        result = minimize(nll, x0, method="L-BFGS-B",
                          options={"maxiter": 500})
        self.weights = result.x[:n_classes]
        self.bias = result.x[n_classes:]
        return self

    def transform(self, proba):
        return _softmax(_safe_log(sanitize_proba(proba, len(self.weights)))
                        * self.weights + self.bias)

    @property
    def n_parameters(self):
        return 0 if self.weights is None else 2 * len(self.weights)

    def describe(self):
        return f"vecteur ({self.n_parameters} parametres)"


class IsotonicOvR(Calibrator):
    """Regression isotone un-contre-tous, puis renormalisation."""

    name = "isotonic"

    def __init__(self):
        self.models = []
        self.classes = None

    def fit(self, proba, y, classes):
        proba = sanitize_proba(proba, len(classes))
        self.classes = list(classes)
        self.models = []
        for i, label in enumerate(classes):
            target = (y == label).astype(float)
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0,
                                       y_max=1.0)
            if target.sum() < 2:          # classe absente de la validation
                self.models.append(None)
                continue
            model.fit(proba[:, i], target)
            self.models.append(model)
        return self

    def transform(self, proba):
        proba = sanitize_proba(proba, len(self.models))
        out = np.zeros_like(proba)
        for i, model in enumerate(self.models):
            out[:, i] = proba[:, i] if model is None else model.predict(proba[:, i])
        total = out.sum(axis=1, keepdims=True)
        # une ligne entierement ecrasee retombe sur les probabilites d'origine
        flat = total[:, 0] <= EPS
        out[flat] = proba[flat]
        total = np.maximum(out.sum(axis=1, keepdims=True), EPS)
        return out / total

    @property
    def n_parameters(self):
        return sum(len(getattr(m, "X_thresholds_", [])) for m in self.models
                   if m is not None)

    def describe(self):
        return f"isotone ({len(self.models)} modeles un-contre-tous)"


def build_calibrators() -> list[Calibrator]:
    return [Identity(), TemperatureScaling(), VectorScaling(), IsotonicOvR()]


# ---------------------------------------------------------------------------
# metriques de calibration
# ---------------------------------------------------------------------------

def expected_calibration_error(proba: np.ndarray, y: np.ndarray, classes,
                               n_bins: int = 15) -> tuple[float, float, pd.DataFrame]:
    """ECE, MCE et le detail par tranche de confiance.

    La confiance est la probabilite de la classe predite. Dans chaque tranche
    on compare cette confiance moyenne a l'exactitude observee.
    """
    classes = np.asarray(classes)
    confidence = proba.max(axis=1)
    predicted = classes[proba.argmax(axis=1)]
    correct = (predicted == y).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows, ece, mce = [], 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        avg_conf = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        weight = mask.sum() / len(y)
        gap = abs(avg_conf - accuracy)
        ece += weight * gap
        mce = max(mce, gap)
        rows.append({"bin_low": round(lo, 3), "bin_high": round(hi, 3),
                     "n_rows": int(mask.sum()), "confidence": round(avg_conf, 4),
                     "accuracy": round(accuracy, 4), "gap": round(gap, 4)})
    return float(ece), float(mce), pd.DataFrame(rows)


def classwise_ece(proba: np.ndarray, y: np.ndarray, classes,
                  n_bins: int = 15) -> pd.DataFrame:
    """ECE calculee classe par classe, en un-contre-tous.

    L'ECE globale est dominee par la classe normale (92 % des lignes) : cette
    vue montre si les classes rares sont, elles aussi, honnetes.
    """
    rows = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i, label in enumerate(classes):
        p = proba[:, i]
        target = (y == label).astype(float)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (p > lo) & (p <= hi)
            if not mask.any():
                continue
            ece += (mask.sum() / len(y)) * abs(float(p[mask].mean())
                                               - float(target[mask].mean()))
        rows.append({"label": label, "classwise_ece": round(ece, 5),
                     "support": int(target.sum())})
    return pd.DataFrame(rows)


def scoring_metrics(proba: np.ndarray, y: np.ndarray, classes) -> dict:
    """Regles de score propres : log-vraisemblance et Brier multiclasse."""
    target = _one_hot(y, classes)
    nll = -float((target * _safe_log(proba)).sum() / len(y))
    brier = float(((proba - target) ** 2).sum(axis=1).mean())
    ece, mce, _ = expected_calibration_error(proba, y, classes)
    classes = np.asarray(classes)
    accuracy = float((classes[proba.argmax(axis=1)] == y).mean())
    return {"nll": round(nll, 5), "brier": round(brier, 5),
            "ece": round(ece, 5), "mce": round(mce, 5),
            "accuracy": round(accuracy, 5),
            "mean_confidence": round(float(proba.max(axis=1).mean()), 5)}


# ---------------------------------------------------------------------------
# politique d'abstention
# ---------------------------------------------------------------------------

def coverage_accuracy_curve(proba: np.ndarray, y: np.ndarray, classes,
                            thresholds=None) -> pd.DataFrame:
    """Compromis couverture / exactitude quand on s'abstient sous un seuil.

    En exploitation : au-dessous du seuil, le systeme repond "incertain" plutot
    qu'une classe. On mesure la part de lignes traitees (couverture) et
    l'exactitude sur ces lignes.
    """
    classes = np.asarray(classes)
    confidence = proba.max(axis=1)
    predicted = classes[proba.argmax(axis=1)]
    correct = predicted == y
    thresholds = thresholds if thresholds is not None else np.arange(0.0, 1.0, 0.05)

    rows = []
    for threshold in thresholds:
        keep = confidence >= threshold
        rows.append({
            "threshold": round(float(threshold), 3),
            "coverage": round(float(keep.mean()), 4),
            "accuracy_on_covered": round(float(correct[keep].mean()), 4)
            if keep.any() else np.nan,
            "n_abstained": int((~keep).sum()),
        })
    return pd.DataFrame(rows)


def choose_threshold(curve: pd.DataFrame, target_accuracy: float) -> float:
    """Seuil le plus bas qui atteint l'exactitude visee (couverture maximale)."""
    ok = curve[curve["accuracy_on_covered"] >= target_accuracy]
    return float(ok["threshold"].min()) if len(ok) else 1.0


@dataclass
class CalibrationResult:
    split: str
    method: str
    description: str
    n_parameters: int
    validation: dict = field(default_factory=dict)
    test: dict = field(default_factory=dict)
