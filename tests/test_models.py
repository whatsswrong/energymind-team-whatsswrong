"""Les roles des composants restent etanches."""

from __future__ import annotations

import numpy as np

from src.inference.pipeline import (NORMAL, UNKNOWN_ANOMALY, fuse,
                                    apply_confidence_policy)

CLASSES = np.array(["normal", "sensor_bias", "sensor_drift"])


def _signals(lgbm_pred, p_max, p_best_anom, best_anom, evidence):
    return {
        "classes": CLASSES,
        "lgbm_pred": np.array(lgbm_pred),
        "p_max": np.array(p_max),
        "p_best_anom": np.array(p_best_anom),
        "best_anom_class": np.array(best_anom),
        "evidence": np.array(evidence),
    }


def test_promotion_sur_preuve():
    """LightGBM dit normal, les preuves disent anomalie -> promotion."""
    signals = _signals(["normal"], [0.9], [0.4], ["sensor_drift"], [True])
    assert fuse(signals, 0.15, 0.60)[0] == "sensor_drift"


def test_promotion_sans_preuve_refusee():
    signals = _signals(["normal"], [0.9], [0.4], ["sensor_drift"], [False])
    assert fuse(signals, 0.15, 0.60)[0] == NORMAL


def test_retrogradation_sans_preuve():
    """Anomalie peu confiante et sans preuve -> retour a normal."""
    signals = _signals(["sensor_bias"], [0.4], [0.4], ["sensor_bias"], [False])
    assert fuse(signals, 0.15, 0.60)[0] == NORMAL


def test_unknown_ne_devine_pas_de_classe():
    """En mode UNKNOWN, aucune classe n'est inventee lors d'une promotion."""
    signals = _signals(["normal"], [0.9], [0.4], ["sensor_drift"], [True])
    result = fuse(signals, 0.15, 0.60, emit_unknown=True)[0]
    assert result == UNKNOWN_ANOMALY
    assert result not in set(CLASSES)


def test_etiquette_non_tronquee():
    """Le dtype objet evite la troncature d'une etiquette longue."""
    signals = _signals(["normal"], [0.9], [0.4], ["sensor_drift"], [True])
    assert len(fuse(signals, 0.15, 0.60, emit_unknown=True)[0]) == len(
        UNKNOWN_ANOMALY)


def test_isolation_forest_ne_renvoie_pas_de_classe():
    """Isolation Forest expose un score numerique, jamais un nom de classe."""
    from src.models.isolation_forest import TrainedIsolationForest
    attributs = set(TrainedIsolationForest.__annotations__)
    assert "classes" not in attributs
    assert {"score", "is_anomalous"} <= set(dir(TrainedIsolationForest))


def test_lightgbm_expose_ses_classes():
    from src.models.lightgbm_model import TrainedLightGBM
    assert "classes" in TrainedLightGBM.__annotations__
    assert {"predict", "predict_proba"} <= set(dir(TrainedLightGBM))


def test_politique_de_confiance():
    """Sous le seuil, le diagnostic devient INCERTAIN."""
    import pandas as pd
    predictions = pd.DataFrame({"y_pred": ["sensor_bias", "overload"]})
    out = apply_confidence_policy(predictions, np.array([0.62, 0.95]), 0.80)
    assert out["reported_label"].tolist() == ["INCERTAIN", "overload"]
    assert out["status"].tolist() == ["INCERTAIN", "DIAGNOSTIC"]


# ---------------------------------------------------------------------------
# Bug 01 : le code source des modeles doit etre suivi par Git
# ---------------------------------------------------------------------------

def test_src_models_suivi_par_git():
    """Un motif `models/` non ancre ignorerait aussi le code source."""
    import subprocess
    sources = ["src/models/lightgbm_model.py", "src/models/isolation_forest.py",
               "src/models/calibration.py"]
    for path in sources:
        result = subprocess.run(["git", "check-ignore", path],
                                capture_output=True, text=True)
        assert result.returncode != 0, f"{path} est ignore par Git"

    tracked = subprocess.run(["git", "ls-files", "src/models"],
                             capture_output=True, text=True).stdout
    for path in sources:
        assert path in tracked, f"{path} n'est pas suivi"


def test_artefact_de_modele_toujours_ignore():
    """Le joblib entraine (~90 Mo) ne doit pas entrer dans l'historique."""
    import subprocess
    result = subprocess.run(
        ["git", "check-ignore", "models/energymind_final_model.joblib"],
        capture_output=True, text=True)
    assert result.returncode == 0, "l'artefact de modele n'est plus ignore"
