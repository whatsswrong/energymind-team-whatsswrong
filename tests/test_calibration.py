"""La calibration produit des probabilites valides, ajustee sur la validation."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.calibration import (VectorScaling, build_calibrators,
                                    choose_threshold,
                                    coverage_accuracy_curve,
                                    expected_calibration_error,
                                    scoring_metrics)

CLASSES = ["normal", "sensor_bias", "sensor_drift"]


@pytest.fixture
def jeu():
    """Jeu synthetique NETTEMENT sur-confiant, comme un modele non calibre.

    On tire la verite depuis des probabilites honnetes, puis on durcit les
    probabilites annoncees (temperature < 1). Le modele annonce donc beaucoup
    plus de certitude qu'il n'en a : exactement ce que la calibration doit
    corriger.
    """
    rng = np.random.default_rng(42)
    logits = rng.normal(size=(4000, 3))
    honest = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array(CLASSES)[[rng.choice(3, p=row) for row in honest]]
    sharp = np.exp(logits / 0.25)
    overconfident = sharp / sharp.sum(axis=1, keepdims=True)
    return overconfident, y


def test_probabilites_valides(jeu):
    proba, y = jeu
    calibrated = VectorScaling().fit(proba, y, CLASSES).transform(proba)
    assert np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0)
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)


def test_ajustement_sur_validation_seulement(jeu):
    """Le calibrateur ne voit que les entrees qu'on lui passe."""
    proba, y = jeu
    validation, test = slice(0, 2000), slice(2000, None)
    calibrator = VectorScaling().fit(proba[validation], y[validation], CLASSES)
    poids = calibrator.weights.copy()
    calibrator.transform(proba[test])          # transformer ne reajuste rien
    assert np.array_equal(poids, calibrator.weights)


def test_calibration_reduit_l_ece(jeu):
    """Ajustee sur la validation, la calibration doit ameliorer le TEST."""
    proba, y = jeu
    validation, test = slice(0, 2000), slice(2000, None)
    calibrator = VectorScaling().fit(proba[validation], y[validation], CLASSES)
    before, _, _ = expected_calibration_error(proba[test], y[test], CLASSES)
    after, _, _ = expected_calibration_error(calibrator.transform(proba[test]),
                                             y[test], CLASSES)
    assert after < before, f"ECE {before:.4f} -> {after:.4f}"


def test_methodes_disponibles():
    noms = [c.name for c in build_calibrators()]
    assert noms == ["aucune", "temperature", "vector", "isotonic"]


def test_seuil_abstention(jeu):
    """Le seuil 0,80 doit effectivement retirer des lignes peu sures."""
    proba, y = jeu
    curve = coverage_accuracy_curve(proba, y, CLASSES)
    at_threshold = curve[np.isclose(curve["threshold"], 0.80)]
    assert len(at_threshold) == 1
    assert 0.0 < float(at_threshold["coverage"].iloc[0]) <= 1.0
    # un seuil plus haut ne peut pas couvrir davantage de lignes
    assert curve["coverage"].is_monotonic_decreasing


def test_metriques_de_score(jeu):
    proba, y = jeu
    stats = scoring_metrics(proba, y, CLASSES)
    assert set(stats) >= {"nll", "brier", "ece", "mce", "accuracy"}
    assert stats["nll"] > 0 and 0.0 <= stats["ece"] <= 1.0
