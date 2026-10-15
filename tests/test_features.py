"""Les 24 features finales : nombre, noms, ordre et valeurs exploitables."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.base_features import BASE_FEATURE_NAMES
from src.features.baseline_features import BASELINE_FEATURE_NAMES
from src.features.persistence_features import PERSISTENCE_FEATURE_NAMES
from src.features.feature_pipeline import (FINAL_FEATURES, N_FINAL_FEATURES,
                                           FeatureBuilder)


@pytest.fixture(scope="module")
def built(cfg, merged, splits):
    table, wide = merged
    builder = FeatureBuilder(wide, cfg)
    builder.fit(splits["PRIMARY"].steps["train"])
    return builder, builder.transform()


def test_exactement_24_features():
    assert N_FINAL_FEATURES == 24
    assert len(FINAL_FEATURES) == 24


def test_composition_des_groupes():
    assert len(BASE_FEATURE_NAMES) == 12
    assert len(BASELINE_FEATURE_NAMES) == 4
    assert len(PERSISTENCE_FEATURE_NAMES) == 8
    assert (BASE_FEATURE_NAMES + BASELINE_FEATURE_NAMES
            + PERSISTENCE_FEATURE_NAMES) == FINAL_FEATURES


def test_ordre_des_colonnes_stable(built):
    _, X = built
    assert list(X.columns) == FINAL_FEATURES


def test_aucune_valeur_infinie(built):
    _, X = built
    assert not np.isinf(X.to_numpy(dtype=float)).any()


def test_familles_rejetees_absentes():
    """Les familles ecartees ne doivent pas revenir dans le pipeline final."""
    rejected = ["drift_slope_6h", "drift_slope_24h", "drift_slope_48h",
                "rolling_mean_dev_6h", "rolling_std_24h"]
    assert not set(rejected) & set(FINAL_FEATURES)


def test_imputation_complete(built, splits):
    builder, X = built
    builder.fit_imputation(X, splits["PRIMARY"].rows["train"])
    assert not builder.impute(X).isna().any().any()


def test_signaux_de_regles_hors_matrice(built):
    """Les signaux de redondance ne doivent jamais etre des features."""
    builder, X = built
    signals = builder.rule_signals()
    assert not set(signals.columns) & set(X.columns)
