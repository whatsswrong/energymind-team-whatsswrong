"""EnergyMind Phase 1 — entrainement et evaluation du modele final.

    python main.py                 pipeline complet sur les 5 decoupages
    python main.py --quick         decoupage principal seul (mise au point)
    python main.py --no-figures    sans generer les images

Architecture finale : LightGBM + Isolation Forest + regles physiques,
24 features, dataset fige de 240 jours.

Enchainement :
    1. chargement et inspection des fichiers
    2. assemblage maison + zone + transformateur, jointures verifiees
    3. decoupages temporels, sans fuite d'evenement
    4. pour chaque decoupage : 24 features -> entrainement -> evaluation
    5. agregation, figures, sauvegarde du modele
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.loader import (inspect_files, expected_shapes, label_summary,
                             load_raw_data)
from src.data.merge import merge_network_data
from src.data.splits import build_all_splits, split_class_report
from src.evaluation import metrics as M
from src.evaluation import plots
from src.features.feature_pipeline import FINAL_FEATURES, feature_report
from src.inference.pipeline import EnergyMindModel, prepare_split

SEPARATOR = "=" * 78


def banner(title: str) -> None:
    print(f"\n{SEPARATOR}\n {title}\n{SEPARATOR}", flush=True)


def save_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)
    print(f"    ecrit : {path.name}  ({len(frame)} lignes)", flush=True)


def evaluate_split(cfg, table, wide, split, raw) -> dict:
    """Entraine le modele sur un decoupage et evalue sa periode de test."""
    prepared, builder = prepare_split(cfg, table, wide, split, raw)
    model = EnergyMindModel(cfg=cfg).fit(prepared)

    test_rows = split.rows["test"]
    predictions = model.predict(prepared, test_rows)
    labels = sorted(pd.unique(prepared.y).tolist())

    summary = M.classification_metrics(predictions["y_true"].to_numpy(),
                                       predictions["y_pred"].to_numpy(), labels)
    events = M.event_level_metrics(predictions,
                                   int(cfg.get("grid.interval_minutes")))
    summary.update(M.event_level_summary(events))
    summary.update({
        "split": split.name,
        "n_features": len(prepared.features),
        "train_time_s": round(model.fit_time_s, 3),
        "inference_time_s": round(model.inference_time_s, 4),
        "promote_min_prob": model.fusion.promote_min_prob,
        "demote_max_prob": model.fusion.demote_max_prob,
        "validation_macro_f1": round(model.fusion.validation_macro_f1, 4),
    })

    per_class = M.per_class_metrics(predictions["y_true"].to_numpy(),
                                    predictions["y_pred"].to_numpy(), labels)
    per_class.insert(0, "split", split.name)

    print(f"    {split.name:<8} macro F1={summary['macro_f1']:.4f}  "
          f"rappel anomalie={summary['anomaly_recall']:.4f}  "
          f"FPR={summary['false_positive_rate']:.4f}  "
          f"evenements={summary['event_recall']:.4f}", flush=True)

    return {"summary": summary, "per_class": per_class, "events": events,
            "predictions": predictions, "prepared": prepared,
            "builder": builder, "model": model, "labels": labels}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="EnergyMind Phase 1")
    parser.add_argument("--config", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="decoupage principal seul")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    np.random.seed(cfg.random_state)

    # ------------------------------------------------------------- 1
    banner("1. CHARGEMENT ET INSPECTION")
    raw = load_raw_data(cfg.data_dir)
    print(inspect_files(raw).to_string(index=False))
    shapes = expected_shapes(raw)
    print("\nDimensions attendues :")
    print(shapes.to_string(index=False))
    assert bool(shapes["match"].all()), "dimensions inattendues"
    save_table(inspect_files(raw), cfg.output("metrics", "data_inspection.csv"))
    save_table(label_summary(raw), cfg.output("metrics",
                                              "label_distribution.csv"))

    # ------------------------------------------------------------- 2
    banner("2. ASSEMBLAGE DE LA TABLE DE TRAVAIL")
    table, wide = merge_network_data(raw)
    print(f"  table maison : {len(table):,} lignes")
    print(f"  vue large    : T={wide.T}, H={wide.H}, Z={wide.Z}")

    # ------------------------------------------------------------- 3
    banner("3. DECOUPAGES TEMPORELS")
    splits = build_all_splits(cfg, table, wide.timestamps)
    for split in splits.values():
        print(f"  {split.describe()}")
    save_table(pd.concat([split_class_report(s, table) for s in splits.values()],
                         ignore_index=True),
               cfg.output("metrics", "split_class_distribution.csv"))

    # ------------------------------------------------------------- 4
    banner("4. ENTRAINEMENT ET EVALUATION DU MODELE FINAL")
    primary_name = cfg.get("splits.primary.name")
    chosen = [primary_name] if args.quick else list(splits)
    results = {name: evaluate_split(cfg, table, wide, splits[name], raw)
               for name in chosen}
    primary = results[primary_name]

    print("\n  Controle de sante des 24 features :")
    report = feature_report(primary["prepared"].X)
    print(report.to_string(index=False))
    assert report["infinite"].sum() == 0, "features infinies"
    save_table(report, cfg.output("metrics", "feature_report.csv"))

    # ------------------------------------------------------------- 5
    banner("5. RESULTATS AGREGES")
    per_split = pd.DataFrame([r["summary"] for r in results.values()])
    per_class = pd.concat([r["per_class"] for r in results.values()],
                          ignore_index=True)

    headline = ["macro_f1", "macro_recall", "macro_precision", "weighted_f1",
                "accuracy", "anomaly_recall", "false_positive_rate",
                "event_recall"]
    aggregated = (per_split[headline].agg(["mean", "std", "min", "max"])
                  .T.round(4).reset_index().rename(columns={"index": "metric"}))
    print(aggregated.to_string(index=False))
    print("\n  F1 par classe (moyenne sur les decoupages) :")
    class_summary = (per_class.groupby("label")["f1"]
                     .agg(["mean", "std"]).round(4)
                     .sort_values("mean", ascending=False))
    print(class_summary.to_string())

    save_table(per_split, cfg.output("metrics", "final_split_results.csv"))
    save_table(per_class, cfg.output("metrics", "final_per_class_metrics.csv"))
    save_table(aggregated, cfg.output("metrics", "final_metrics_summary.csv"))
    save_table(primary["predictions"],
               cfg.output("predictions", "final_predictions.csv"))

    confusion = M.confusion(primary["predictions"]["y_true"].to_numpy(),
                            primary["predictions"]["y_pred"].to_numpy(),
                            primary["labels"])
    confusion.to_csv(cfg.output("metrics", "final_confusion_matrix.csv"))

    events = (primary["events"].groupby("label")
              .agg(event_recall=("detected", "mean"),
                   event_class_recall=("correctly_classified", "mean"),
                   n_events=("detected", "size"),
                   median_latency_steps=("latency_steps", "median"))
              .reset_index())
    save_table(events, cfg.output("metrics", "final_event_level.csv"))

    # ------------------------------------------------------------- 6
    banner("6. SAUVEGARDE DU MODELE ET DES FIGURES")
    model_path = cfg.models_dir / "energymind_final_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"lightgbm": primary["model"].lightgbm,
                 "isolation_forest": primary["model"].isolation_forest,
                 "fusion": primary["model"].fusion,
                 "feature_stats": primary["builder"].stats,
                 "rule_thresholds": primary["prepared"].thresholds,
                 "features": FINAL_FEATURES,
                 "split": primary_name}, model_path)
    print(f"    modele : {model_path.name}")

    if not args.no_figures:
        produced = [
            plots.confusion_matrix_figure(
                confusion, "Matrice de confusion — modele final",
                cfg.output("figures", "final_confusion_matrix.png")),
            plots.per_class_f1_final(
                class_summary, cfg.output("figures", "final_per_class_f1.png")),
            plots.event_recall_final(
                events, cfg.output("figures", "final_event_recall.png")),
        ]
        for path in produced:
            print(f"    {path.name}")

    banner("TERMINE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
