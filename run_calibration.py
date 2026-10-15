"""Calibration de confiance du modele retenu.

    python run_calibration.py             entraine, calibre, evalue, trace
    python run_calibration.py --reuse     reutilise les probabilites sauvegardees

Architecture et features figees : Isolation Forest + LightGBM + regles, jeu de
features EXP_FINAL (24 features). On ne retouche ni le modele ni les features :
on ajuste uniquement la facon dont il exprime sa confiance.

Protocole :
    1. pour chaque decoupage, recuperer les probabilites sur validation ET test
    2. ajuster les calibrateurs sur la VALIDATION seule
    3. choisir la methode par log-vraisemblance de validation
    4. mesurer sur le TEST : ECE, MCE, Brier, fiabilite, abstention
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.loader import load_raw_data
from src.data.merge import merge_network_data
from src.data.splits import build_all_splits
from src.inference.pipeline import EnergyMindModel, prepare_split
from src.models.calibration import (build_calibrators, choose_threshold,
                                    classwise_ece, coverage_accuracy_curve,
                                    expected_calibration_error,
                                    scoring_metrics)

WEAK = ("sensor_bias", "sensor_drift")
TARGET_ACCURACY = 0.95
CONFIDENCE_THRESHOLD_KEY = "inference.confidence_threshold"
SEP = "=" * 78


def banner(title: str) -> None:
    print(f"\n{SEP}\n {title}\n{SEP}", flush=True)


def collect_probabilities(cfg, table, wide, splits, raw, cache: Path) -> dict:
    """Entraine le modele retenu et conserve ses probabilites par decoupage."""
    store = {}
    for name, split in splits.items():
        prepared, _ = prepare_split(cfg, table, wide, split, raw)
        model = EnergyMindModel(cfg=cfg).fit(prepared)
        entry = {"classes": np.asarray(model.lightgbm.classes)}
        for part in ("validation", "test"):
            rows = split.rows[part]
            entry[f"proba_{part}"] = model.lightgbm.predict_proba(
                prepared.X.loc[rows])
            entry[f"y_{part}"] = prepared.y[rows]
            entry[f"fused_{part}"] = model.predict(
                prepared, rows)["y_pred"].to_numpy()
        store[name] = entry
        print(f"    {name:<8} validation {len(entry['y_validation']):,} lignes, "
              f"test {len(entry['y_test']):,} lignes", flush=True)

    np.savez_compressed(cache, **{f"{k}__{field}": value
                                  for k, entry in store.items()
                                  for field, value in entry.items()})
    print(f"    probabilites sauvegardees : {cache.name}")
    return store


def load_probabilities(cache: Path) -> dict:
    data = np.load(cache, allow_pickle=True)
    store: dict = {}
    for key in data.files:
        split, field = key.split("__", 1)
        store.setdefault(split, {})[field] = data[key]
    return store


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Calibration de confiance")
    ap.add_argument("--reuse", action="store_true",
                    help="reutiliser les probabilites deja sauvegardees")
    args = ap.parse_args(argv)

    cfg = load_config()
    np.random.seed(cfg.random_state)
    cache = cfg.output("datasets", "calibration_probabilities.npz")

    banner("1. PROBABILITES DU MODELE RETENU (24 features, approche B)")
    if args.reuse and cache.exists():
        store = load_probabilities(cache)
        print(f"    reutilisation de {cache.name}")
    else:
        raw = load_raw_data(cfg.data_dir)
        table, wide = merge_network_data(raw)
        splits = build_all_splits(cfg, table, wide.timestamps)
        store = collect_probabilities(cfg, table, wide, splits, raw, cache)

    # ------------------------------------------------------------------ 2
    banner("2. AJUSTEMENT SUR LA VALIDATION, MESURE SUR LE TEST")
    rows, reliability, classwise, curves = [], [], [], []
    for split, entry in store.items():
        classes = entry["classes"]
        for calibrator in build_calibrators():
            calibrator.fit(entry["proba_validation"], entry["y_validation"],
                           classes)
            val = calibrator.transform(entry["proba_validation"])
            test = calibrator.transform(entry["proba_test"])

            val_metrics = scoring_metrics(val, entry["y_validation"], classes)
            test_metrics = scoring_metrics(test, entry["y_test"], classes)
            rows.append({"split": split, "method": calibrator.name,
                         "description": calibrator.describe(),
                         "n_parameters": calibrator.n_parameters,
                         **{f"val_{k}": v for k, v in val_metrics.items()},
                         **{f"test_{k}": v for k, v in test_metrics.items()}})

            _, _, bins = expected_calibration_error(test, entry["y_test"],
                                                    classes)
            bins.insert(0, "method", calibrator.name)
            bins.insert(0, "split", split)
            reliability.append(bins)

            cw = classwise_ece(test, entry["y_test"], classes)
            cw.insert(0, "method", calibrator.name)
            cw.insert(0, "split", split)
            classwise.append(cw)

            curve = coverage_accuracy_curve(test, entry["y_test"], classes)
            curve.insert(0, "method", calibrator.name)
            curve.insert(0, "split", split)
            curves.append(curve)
        print(f"    {split} : 4 methodes evaluees", flush=True)

    results = pd.DataFrame(rows)
    reliability = pd.concat(reliability, ignore_index=True)
    classwise = pd.concat(classwise, ignore_index=True)
    curves = pd.concat(curves, ignore_index=True)

    summary = (results.groupby("method")
               [["val_nll", "test_nll", "test_ece", "test_mce", "test_brier",
                 "test_accuracy", "test_mean_confidence", "n_parameters"]]
               .mean().round(5).reset_index())
    summary = summary.sort_values("val_nll")
    print()
    print(summary.to_string(index=False))

    # ------------------------------------------------------------------ 3
    banner("3. METHODE RETENUE")
    best = summary.iloc[0]["method"]
    base = summary[summary.method == "aucune"].iloc[0]
    chosen = summary[summary.method == best].iloc[0]
    print(f"  Choisie par log-vraisemblance de VALIDATION : {best}")
    print(f"    ECE   : {base.test_ece:.4f}  ->  {chosen.test_ece:.4f}  "
          f"({100 * (1 - chosen.test_ece / max(base.test_ece, 1e-9)):.0f} % de reduction)")
    print(f"    MCE   : {base.test_mce:.4f}  ->  {chosen.test_mce:.4f}")
    print(f"    Brier : {base.test_brier:.4f}  ->  {chosen.test_brier:.4f}")
    print(f"    Exactitude : {base.test_accuracy:.4f}  ->  "
          f"{chosen.test_accuracy:.4f}  (inchangee si temperature)")
    print(f"    Confiance moyenne annoncee : {base.test_mean_confidence:.4f}"
          f"  ->  {chosen.test_mean_confidence:.4f}")
    for _, row in results[results.method == best].iterrows():
        print(f"      {row.split:<8} {row.description}")

    # ------------------------------------------------------------------ 4
    banner("4. CALIBRATION PAR CLASSE (classes faibles comprises)")
    cw = (classwise[classwise.method.isin(["aucune", best])]
          .groupby(["label", "method"])["classwise_ece"].mean()
          .unstack().round(5))
    cw["gain"] = (cw["aucune"] - cw[best]).round(5)
    print(cw.sort_values("aucune", ascending=False).to_string())

    # ------------------------------------------------------------------ 5
    banner("5. POLITIQUE D'ABSTENTION")
    curve = (curves[curves.method == best]
             .groupby("threshold")[["coverage", "accuracy_on_covered"]]
             .mean().reset_index())
    threshold = choose_threshold(curve, TARGET_ACCURACY)
    at = curve[curve.threshold == threshold]
    print(f"  Pour garantir {TARGET_ACCURACY:.0%} d'exactitude sur les lignes "
          f"traitees :")
    if len(at):
        print(f"    seuil de confiance {threshold:.2f}  ->  couverture "
              f"{float(at.coverage.iloc[0]):.1%}, exactitude "
              f"{float(at.accuracy_on_covered.iloc[0]):.1%}")
        print(f"    soit {100 * (1 - float(at.coverage.iloc[0])):.1f} % des "
              "lignes renvoyees comme INCERTAIN")
    print()
    print(curve[curve.threshold.isin([0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])]
          .round(4).to_string(index=False))

    # ------------------------------------------------------------------ 6
    banner("6. ECRITURE DES RESULTATS")
    for frame, name in ((results, "calibration_results.csv"),
                        (summary, "calibration_summary.csv"),
                        (reliability, "calibration_reliability_bins.csv"),
                        (classwise, "calibration_classwise_ece.csv"),
                        (curves, "calibration_coverage_curves.csv")):
        path = cfg.output("metrics", name)
        frame.to_csv(path, index=False)
        print(f"    {name}  ({len(frame)} lignes)")

    from src.evaluation import plots as CF
    for path in CF.make_all(cfg, store, results, reliability, classwise,
                            curves, best, threshold):
        print(f"    {path.name}")

    banner("TERMINE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
