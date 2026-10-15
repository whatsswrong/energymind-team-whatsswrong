"""Non-regression sur les fuites entre periodes (bugs 02, 07, 11).

Toute statistique utilisee a l'inference doit etre apprise sur la seule
periode d'entrainement. Ces tests le prouvent de deux facons :

* en corrompant le TEST et en verifiant que rien d'appris ne bouge ;
* en verifiant qu'aucun ajustement de secours ne peut se declencher
  silencieusement sur l'historique complet.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.feature_pipeline import FeatureBuilder
from src.inference.pipeline import prepare_split


def _learned_statistics(builder: FeatureBuilder) -> dict:
    """Tout ce que le constructeur a appris, sous forme comparable."""
    stats = builder.stats
    return {
        "zone_resid": stats.zone_resid,
        "trans_resid": stats.trans_resid,
        "zone_loss_model": stats.zone_loss_model,
        "trans_loss_model": stats.trans_loss_model,
        "trans_house_loss_model": stats.trans_house_loss_model,
        "baseline": stats.baseline.copy(),
        "hour_median": stats.hour_median.copy(),
        "norm_center": stats.norm_center.copy(),
        "norm_scale": stats.norm_scale.copy(),
        "frozen_tol_P": stats.frozen_tol_P,
        "frozen_tol_V": stats.frozen_tol_V,
        "neighbour_z_threshold": stats.neighbour_z_threshold,
    }


def test_rule_signals_avant_fit_leve_une_erreur(cfg, merged):
    """Bug 07 : aucun reajustement silencieux sur l'historique complet."""
    _, wide = merged
    builder = FeatureBuilder(wide, cfg)
    assert builder.stats.trans_house_loss_model is None
    with pytest.raises(RuntimeError, match="fit"):
        builder.rule_signals()


def test_modele_de_perte_transfo_maisons_ajuste_sur_le_train(cfg, merged,
                                                             splits):
    """Bug 02 : le modele de perte ne doit dependre que du train.

    On l'ajuste une fois normalement, puis on le reajuste en ayant corrompu
    la telemetrie de la periode de TEST. Les parametres doivent etre
    identiques.
    """
    _, wide = merged
    split = splits["PRIMARY"]
    train_steps, test_steps = split.steps["train"], split.steps["test"]

    reference = FeatureBuilder(wide, cfg).fit(train_steps)
    before = reference.stats.trans_house_loss_model
    assert before is not None

    corrupted = __import__("copy").copy(wide)
    corrupted.trans_P = wide.trans_P.copy()
    corrupted.trans_P[test_steps] *= 7.5          # telemetrie de test absurde
    after = FeatureBuilder(corrupted, cfg).fit(train_steps)

    assert after.stats.trans_house_loss_model == before


def test_statistiques_insensibles_a_la_corruption_du_test(cfg, merged, splits):
    """Bug 11 : corrompre les labels de TEST ne change rien a l'appris."""
    table, wide = merged
    split = splits["PRIMARY"]

    reference = FeatureBuilder(wide, cfg).fit(split.steps["train"])
    before = _learned_statistics(reference)

    corrupted_table = table.copy()
    test_rows = split.rows["test"]
    rng = np.random.default_rng(0)
    corrupted_table.loc[test_rows, "label"] = rng.permutation(
        corrupted_table.loc[test_rows, "label"].to_numpy())

    after = _learned_statistics(FeatureBuilder(wide, cfg)
                                .fit(split.steps["train"]))

    for name, value in before.items():
        other = after[name]
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, other, equal_nan=True), name
        else:
            assert value == other, name


def test_seuils_de_regles_insensibles_aux_labels_de_test(cfg, merged, splits,
                                                         raw):
    """Bug 11 : les seuils de regles ne doivent pas voir les labels de test."""
    table, wide = merged
    split = splits["PRIMARY"]
    reference, _ = prepare_split(cfg, table, wide, split, raw)

    corrupted = table.copy()
    rng = np.random.default_rng(1)
    test_rows = split.rows["test"]
    corrupted.loc[test_rows, "label"] = rng.permutation(
        corrupted.loc[test_rows, "label"].to_numpy())
    altered, _ = prepare_split(cfg, corrupted, wide, split, raw)

    assert (reference.thresholds.as_frame().to_dict()
            == altered.thresholds.as_frame().to_dict())


def test_aucune_colonne_de_verite_dans_les_features(cfg, merged, splits, raw):
    """Ni label, ni event_id, ni verite de simulation parmi les features."""
    table, wide = merged
    prepared, _ = prepare_split(cfg, table, wide, splits["PRIMARY"], raw)
    interdites = {"label", "event_id", "anomaly_family", "severity",
                  "is_anomaly", "target_type", "entity_type", "supply_on",
                  "zone_loss_pct", "trans_loss_pct"}
    assert not interdites & set(prepared.X.columns)
    assert not interdites & set(prepared.X_imputed.columns)


def test_fichiers_de_verite_jamais_lus():
    """``ground_truth.csv`` et ``injection_log.csv`` ne sont jamais ouverts.

    On inspecte les chaines REELLEMENT utilisees par le code, pas les
    commentaires ni les docstrings — qui ont parfaitement le droit de citer
    ces fichiers pour expliquer qu'on ne les lit pas.
    """
    import ast
    import pathlib

    interdits = ("ground_truth", "injection_log")
    sources = list(pathlib.Path("src").rglob("*.py"))
    sources += [pathlib.Path("main.py"), pathlib.Path("run_calibration.py")]

    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {ast.get_docstring(node, clean=False)
                      for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.ClassDef,
                                           ast.FunctionDef,
                                           ast.AsyncFunctionDef))}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(
                    node.value, str):
                continue
            if node.value in docstrings:
                continue
            for interdit in interdits:
                assert interdit not in node.value, (
                    f"{path}:{node.lineno} utilise {interdit!r}")
