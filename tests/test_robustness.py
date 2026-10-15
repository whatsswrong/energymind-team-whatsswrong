"""Non-regression sur la robustesse numerique (bugs 03, 04, 05, 08, 09, 10, 13, 14).

Ces tests portent sur des entrees degradees : coupures de capteur, charge
constante, probabilites invalides, telemetrie dupliquee, classe absente du
train. Chacun reproduit la condition qui faisait echouer silencieusement le
code avant correction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import _assert_unique
from src.features.windows import (fit_loss_model, rolling_mean, rolling_median,
                                  rolling_slope, rolling_std)
from src.models.calibration import (IsotonicOvR, TemperatureScaling,
                                    VectorScaling, sanitize_proba)

CLASSES = ["normal", "sensor_bias", "sensor_drift"]


# ---------------------------------------------------------------------------
# Bug 03 / 10 : fenetres causales pendant et apres une coupure de capteur
# ---------------------------------------------------------------------------

def test_rolling_slope_ne_fabrique_pas_de_pente_pendant_une_coupure():
    """Bug 03 : le report avant ne doit pas compter comme une mesure reelle."""
    values = np.array([[10.0], [11.0], [12.0], [np.nan], [np.nan], [np.nan]])
    slope = rolling_slope(values, window=3, min_periods=3)
    # trois mesures reelles au pas 2 : pente definie
    assert np.isfinite(slope[2, 0])
    # ensuite la fenetre ne contient plus assez de mesures reelles
    assert np.isnan(slope[3, 0]) and np.isnan(slope[4, 0])
    assert np.isnan(slope[5, 0])


def test_rolling_slope_reste_nan_apres_reconnexion():
    """Bug 10 : la reconnexion ne reconstitue pas un historique instantane."""
    values = np.array([[1.0], [2.0], [np.nan], [np.nan], [np.nan], [9.0]])
    slope = rolling_slope(values, window=4, min_periods=3)
    # au pas 5, la fenetre [2..5] ne contient qu'une seule mesure reelle
    assert np.isnan(slope[5, 0])


def test_fenetres_glissantes_sans_fuite_future():
    """Modifier un pas futur ne change aucune valeur passee."""
    base = np.arange(20, dtype=float).reshape(-1, 1)
    altered = base.copy()
    altered[15:] = 999.0
    for function in (rolling_mean, rolling_median, rolling_std):
        reference = function(base, 4, 2)
        modified = function(altered, 4, 2)
        assert np.allclose(reference[:15], modified[:15], equal_nan=True)
    assert np.allclose(rolling_slope(base, 4, 2)[:15],
                       rolling_slope(altered, 4, 2)[:15], equal_nan=True)


def test_fenetres_respectent_min_periods_sur_mesures_reelles():
    values = np.array([[1.0], [np.nan], [np.nan], [np.nan], [5.0]])
    assert np.isnan(rolling_mean(values, 3, 2)[3, 0])
    assert np.isnan(rolling_std(values, 3, 2)[3, 0])


# ---------------------------------------------------------------------------
# Bug 04 / 08 : fit_loss_model face aux entrees degradees
# ---------------------------------------------------------------------------

def test_fit_loss_model_refuse_longueurs_incompatibles():
    with pytest.raises(ValueError, match="longueurs"):
        fit_loss_model(np.zeros(5), np.zeros(4))


def test_fit_loss_model_charge_constante_sans_rankwarning():
    """Bug 04 : polyfit est instable si la charge n'a aucune variance."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        intercept, slope = fit_loss_model(np.full(300, 12.0),
                                          np.full(300, 0.02))
    assert slope == 0.0
    assert np.isfinite(intercept)


def test_fit_loss_model_charge_quasi_constante():
    intercept, slope = fit_loss_model(np.full(300, 12.0) + 1e-12,
                                      np.linspace(0.0, 0.1, 300))
    assert slope == 0.0 and np.isfinite(intercept)


def test_fit_loss_model_cas_limites():
    assert fit_loss_model(np.full(10, np.nan), np.full(10, np.nan)) == (0.0, 0.0)
    assert fit_loss_model(np.array([1.0]), np.array([0.5])) == (0.5, 0.0)
    assert fit_loss_model(np.array([]), np.array([])) == (0.0, 0.0)
    intercept, slope = fit_loss_model(np.array([1.0, np.inf, 3.0]),
                                      np.array([0.1, 0.2, 0.3]))
    assert np.isfinite(intercept) and np.isfinite(slope)


# ---------------------------------------------------------------------------
# Bug 05 / 09 : probabilites extremes ou invalides
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("proba", [
    np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),          # exactement 0 et 1
    np.array([[np.nan, np.inf, -np.inf], [0.0, 0.0, 0.0]]),  # non finis
    np.array([[-0.5, 0.5, 1.0]]),                           # negatifs
    np.array([[2.0, 3.0, 5.0]]),                            # non normalisees
    np.array([[1e-300, 1e-300, 1.0]]),                      # extremement petit
])
def test_sanitize_proba_rend_une_matrice_valide(proba):
    out = sanitize_proba(proba, 3)
    assert np.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()
    assert np.allclose(out.sum(axis=1), 1.0)


def test_sanitize_proba_ligne_nulle_devient_uniforme():
    out = sanitize_proba(np.zeros((1, 4)), 4)
    assert np.allclose(out, 0.25)


def test_sanitize_proba_rejette_une_largeur_incoherente():
    with pytest.raises(ValueError, match="colonnes"):
        sanitize_proba(np.zeros((2, 5)), 3)
    with pytest.raises(ValueError, match="2D"):
        sanitize_proba(np.zeros(5), 5)


@pytest.mark.parametrize("calibrator", [TemperatureScaling, VectorScaling,
                                        IsotonicOvR])
def test_calibration_survit_aux_probabilites_extremes(calibrator):
    """Bug 05 : log(0) ne doit produire ni NaN ni inf."""
    proba = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                      [1e-300, 1e-300, 1.0]])
    y = np.array(["sensor_drift", "normal", "sensor_bias", "sensor_drift"])
    out = calibrator().fit(proba, y, CLASSES).transform(proba)
    assert np.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()
    assert np.allclose(out.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Bug 13 : telemetrie dupliquee
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("keys,frame", [
    (["timestamp", "house_id"],
     pd.DataFrame({"timestamp": [1, 1, 2], "house_id": ["H01", "H01", "H01"]})),
    (["timestamp", "zone_id"],
     pd.DataFrame({"timestamp": [1, 1], "zone_id": ["Z1", "Z1"]})),
    (["timestamp"], pd.DataFrame({"timestamp": [1, 1, 2]})),
])
def test_doublons_de_telemetrie_rejetes(keys, frame):
    with pytest.raises(ValueError, match="duplique"):
        _assert_unique(frame, keys, "fichier_test.csv")


def test_telemetrie_sans_doublon_acceptee():
    frame = pd.DataFrame({"timestamp": [1, 2], "house_id": ["H01", "H01"]})
    _assert_unique(frame, ["timestamp", "house_id"], "ok.csv")


# ---------------------------------------------------------------------------
# Bug 14 : alignement des colonnes de probabilites
# ---------------------------------------------------------------------------

class _ModelePartiel:
    """Modele factice n'ayant vu que deux des trois classes canoniques."""

    classes_ = np.array(["normal", "sensor_drift"])

    def predict_proba(self, X):
        return np.array([[0.3, 0.7], [0.9, 0.1]])


def test_probabilites_alignees_sur_les_classes_canoniques():
    from src.models.lightgbm_model import TrainedLightGBM
    trained = TrainedLightGBM(model=_ModelePartiel(), classes=CLASSES,
                              train_time_s=0.0, n_train_rows=2,
                              features=["f"])
    X = pd.DataFrame({"f": [0.0, 1.0]})
    proba = trained.predict_proba(X)

    assert proba.shape == (2, 3)
    # la classe absente du train recoit une probabilite nulle
    assert np.allclose(proba[:, CLASSES.index("sensor_bias")], 0.0)
    assert np.allclose(proba[:, CLASSES.index("normal")], [0.3, 0.9])
    assert np.allclose(proba[:, CLASSES.index("sensor_drift")], [0.7, 0.1])
    # argmax doit designer la bonne etiquette
    assert list(np.asarray(CLASSES)[proba.argmax(axis=1)]) == [
        "sensor_drift", "normal"]


def test_classe_inconnue_du_vocabulaire_canonique_rejetee():
    from src.models.lightgbm_model import TrainedLightGBM

    class _ModeleIncoherent(_ModelePartiel):
        classes_ = np.array(["normal", "classe_inexistante"])

    trained = TrainedLightGBM(model=_ModeleIncoherent(), classes=CLASSES,
                              train_time_s=0.0, n_train_rows=2,
                              features=["f"])
    with pytest.raises(ValueError, match="absente"):
        trained.predict_proba(pd.DataFrame({"f": [0.0, 1.0]}))
