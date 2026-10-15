"""Metriques d'evaluation : classes connues, normal/anomalie, niveau evenement."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_fscore_support,
                             precision_score, recall_score)

NORMAL = "normal"


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           labels: list[str] | None = None) -> dict:
    """Metriques globales. L'exactitude est volontairement secondaire."""
    labels = labels or sorted(set(y_true) | set(y_pred))
    true_anom = y_true != NORMAL
    pred_anom = y_pred != NORMAL
    n_normal = int((~true_anom).sum())

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro",
                                                 zero_division=0, labels=labels)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0, labels=labels)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0, labels=labels)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0, labels=labels)),
        "anomaly_recall": float((pred_anom & true_anom).sum() / max(true_anom.sum(), 1)),
        "false_positive_rate": float((pred_anom & ~true_anom).sum() / max(n_normal, 1)),
        "specificity": float((~pred_anom & ~true_anom).sum() / max(n_normal, 1)),
        "n_rows": int(len(y_true)),
        "n_anomaly_rows": int(true_anom.sum()),
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      labels: list[str] | None = None) -> pd.DataFrame:
    labels = labels or sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"label": labels, "precision": precision,
                         "recall": recall, "f1": f1, "support": support})


def confusion(y_true: np.ndarray, y_pred: np.ndarray,
              labels: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def event_level_metrics(predictions: pd.DataFrame,
                        step_minutes: int = 15) -> pd.DataFrame:
    """Un evenement long ne doit pas peser plus qu'un evenement court.

    Un evenement est "detecte" si AU MOINS une de ses observations est predite
    non normale. La latence est le nombre de pas entre le debut de l'evenement
    et la premiere detection.
    """
    rows = []
    with_event = predictions[predictions["event_id"].notna()]
    for event_id, grp in with_event.groupby("event_id"):
        grp = grp.sort_values("timestamp")
        detected_mask = (grp["y_pred"] != NORMAL).to_numpy()
        correct_mask = (grp["y_pred"] == grp["y_true"]).to_numpy()
        first = int(np.argmax(detected_mask)) if detected_mask.any() else -1
        rows.append({
            "event_id": event_id,
            "label": grp["y_true"].iloc[0],
            "n_rows": len(grp),
            "detected": bool(detected_mask.any()),
            "correctly_classified": bool(correct_mask.any()),
            "latency_steps": first if first >= 0 else np.nan,
            "latency_minutes": first * step_minutes if first >= 0 else np.nan,
        })
    return pd.DataFrame(rows)


def event_level_summary(events: pd.DataFrame) -> dict:
    if events.empty:
        return {"event_recall": np.nan, "event_class_recall": np.nan,
                "median_latency_steps": np.nan, "n_events": 0}
    return {
        "event_recall": float(events["detected"].mean()),
        "event_class_recall": float(events["correctly_classified"].mean()),
        "median_latency_steps": float(events.loc[events["detected"],
                                                 "latency_steps"].median()),
        "n_events": int(len(events)),
    }


def unknown_anomaly_metrics(predictions: pd.DataFrame,
                            hidden_class: str) -> dict:
    """Detection d'une classe jamais vue a l'entrainement.

    On ne mesure PAS la bonne classe (impossible : elle n'existe pas pour le
    modele) mais la capacite a signaler la ligne comme non normale.
    """
    hidden = predictions["y_true"] == hidden_class
    truly_normal = predictions["y_true"] == NORMAL
    flagged = predictions["y_pred"] != NORMAL

    events = predictions.loc[hidden & predictions["event_id"].notna()]
    detected_events = (events.assign(flag=flagged[hidden & predictions["event_id"].notna()])
                       .groupby("event_id")["flag"].any())

    return {
        "hidden_class": hidden_class,
        "n_hidden_rows": int(hidden.sum()),
        "n_hidden_events": int(detected_events.size),
        "unknown_row_detection_rate": float(flagged[hidden].mean())
        if hidden.any() else np.nan,
        "unknown_event_detection_rate": float(detected_events.mean())
        if detected_events.size else np.nan,
        "normal_false_positive_rate": float(flagged[truly_normal].mean())
        if truly_normal.any() else np.nan,
    }


def isolation_forest_score_stats(predictions: pd.DataFrame,
                                 hidden_class: str | None = None) -> pd.DataFrame:
    """Distribution des scores d'Isolation Forest par groupe de verite."""
    if predictions["if_score"].isna().all():
        return pd.DataFrame()
    group = np.where(predictions["y_true"] == NORMAL, "normal",
                     np.where(predictions["y_true"] == hidden_class,
                              "anomalie inconnue", "anomalie connue"))
    frame = predictions.assign(group=group)
    return (frame.groupby("group")["if_score"]
            .agg(["count", "mean", "std", "min",
                  lambda s: s.quantile(0.05), "median",
                  lambda s: s.quantile(0.95), "max"])
            .set_axis(["count", "mean", "std", "min", "p05", "median", "p95", "max"],
                      axis=1)
            .reset_index())


def higher_level_metrics(flags: pd.DataFrame, labels: pd.DataFrame,
                         entity_col: str, flag_col: str,
                         fault_labels: tuple) -> dict:
    """Detection par REGLES des defauts de compteur zone / transformateur.

    Ces classes ne sont pas apprises par LightGBM (les maisons ne sont pas
    labellisees quand seul un compteur de niveau superieur ment) : elles sont
    evaluees ici, sur la couche de regles.
    """
    keys = ["timestamp"] + ([entity_col] if entity_col else [])
    merged = flags.merge(labels, on=keys, how="inner", validate="one_to_one")
    truth = merged["label"].isin(fault_labels)
    pred = merged[flag_col].astype(bool)
    normal = merged["label"] == NORMAL
    return {
        "n_rows": int(len(merged)),
        "n_fault_rows": int(truth.sum()),
        "recall": float((pred & truth).sum() / max(truth.sum(), 1)),
        "precision": float((pred & truth).sum() / max(pred.sum(), 1)),
        "false_positive_rate": float((pred & normal).sum() / max(normal.sum(), 1)),
    }
