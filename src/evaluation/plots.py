"""Figures de documentation. Toutes tracees a partir des resultats reels.

Chaque figure est accompagnee du CSV des valeurs qu'elle represente, pour que
les chiffres soient reutilisables dans le rapport sans les relire sur l'image.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 160
COLOR_A = "#2E6F9E"
COLOR_B = "#C8622B"
plt.rcParams.update({
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.size": 10,
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def confusion_matrix_figure(matrix: pd.DataFrame, title: str,
                            path: Path) -> Path:
    """Matrice de confusion normalisee par ligne (recall par classe)."""
    totals = matrix.to_numpy().sum(axis=1, keepdims=True)
    normalised = np.divide(matrix.to_numpy(), np.maximum(totals, 1))

    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Classe predite")
    ax.set_ylabel("Classe reelle")
    ax.set_title(title)
    for i in range(normalised.shape[0]):
        for j in range(normalised.shape[1]):
            count = int(matrix.to_numpy()[i, j])
            if count:
                ax.text(j, i, f"{normalised[i, j]:.2f}\n({count})",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if normalised[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, label="Proportion de la classe reelle")
    ax.grid(False)
    return _save(fig, path)


def approach_comparison_figure(summary: pd.DataFrame, path: Path) -> Path:
    """Barres groupees : Macro F1, Macro Recall, taux de faux positifs."""
    metrics = ["macro_f1", "macro_recall", "false_positive_rate"]
    pretty = ["Macro F1", "Macro Recall", "Taux de faux positifs"]
    a = [float(summary.loc[summary.approach == "A", m].iloc[0]) for m in metrics]
    b = [float(summary.loc[summary.approach == "B", m].iloc[0]) for m in metrics]

    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.19, a, 0.38, label="A : LightGBM + regles", color=COLOR_A)
    ax.bar(x + 0.19, b, 0.38, label="B : + Isolation Forest", color=COLOR_B)
    for xi, (va, vb) in enumerate(zip(a, b)):
        ax.text(xi - 0.19, va + 0.012, f"{va:.3f}", ha="center", fontsize=9)
        ax.text(xi + 0.19, vb + 0.012, f"{vb:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(pretty)
    ax.set_ylabel("Valeur")
    ax.set_ylim(0, max(max(a), max(b)) * 1.22 + 0.05)
    ax.set_title("Comparaison des approches sur le test du split principal")
    ax.legend()
    return _save(fig, path)


def temporal_robustness_figure(per_split: pd.DataFrame, path: Path) -> Path:
    """Macro F1 des deux approches sur chaque decoupage temporel."""
    splits = per_split["split"].unique().tolist()
    x = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(9, 5))
    for approach, color in (("A", COLOR_A), ("B", COLOR_B)):
        sub = per_split[per_split.approach == approach].set_index("split")
        values = [float(sub.loc[s, "macro_f1"]) for s in splits]
        ax.plot(x, values, "o-", color=color, linewidth=2, markersize=7,
                label=f"Approche {approach} (moy {np.mean(values):.3f} "
                      f"+/- {np.std(values):.3f})")
        for xi, v in zip(x, values):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_xlabel("Decoupage temporel (periode de test)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Stabilite des approches selon la periode evaluee")
    ax.legend(loc="lower right")
    return _save(fig, path)


def per_class_f1_figure(per_class: pd.DataFrame, path: Path) -> Path:
    """F1 par classe, approche A contre approche B."""
    pivot = per_class.pivot(index="label", columns="approach", values="f1")
    pivot = pivot.sort_values("A")
    y = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(y - 0.19, pivot["A"], 0.38, label="A : LightGBM + regles",
            color=COLOR_A)
    ax.barh(y + 0.19, pivot["B"], 0.38, label="B : + Isolation Forest",
            color=COLOR_B)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("F1")
    ax.set_xlim(0, 1.08)
    ax.set_title("F1 par classe apprise (test du split principal)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def isolation_forest_scores_figure(scores: pd.DataFrame, path: Path) -> Path:
    """Distributions des scores d'Isolation Forest par nature de ligne.

    Un score bas signifie "atypique". Si la courbe des anomalies inconnues se
    decale vers la gauche par rapport au normal, Isolation Forest apporte bien
    une capacite de detection de nouveaute.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = {"normal": "#4C72B0", "anomalie connue": "#55A868",
               "anomalie inconnue": "#C44E52"}
    for group, color in palette.items():
        values = scores.loc[scores["group"] == group, "if_score"].to_numpy()
        values = values[np.isfinite(values)]
        if values.size < 5:
            continue
        ax.hist(values, bins=60, density=True, alpha=0.5, color=color,
                label=f"{group} (n={values.size:,}, med={np.median(values):.3f})")
        ax.axvline(np.median(values), color=color, linestyle="--", linewidth=1.3)
    threshold = scores.attrs.get("threshold")
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle=":", linewidth=1.8,
                   label=f"seuil de decision ({threshold:.3f})")
    ax.set_xlabel("Score Isolation Forest (plus bas = plus atypique)")
    ax.set_ylabel("Densite")
    ax.set_title("Scores Isolation Forest : normal, anomalies connues, inconnues")
    ax.legend(fontsize=9)
    return _save(fig, path)


def event_recall_figure(events: pd.DataFrame, path: Path) -> Path:
    """Rappel au niveau EVENEMENT, pour que les anomalies longues ne dominent pas."""
    pivot = events.pivot(index="label", columns="approach", values="event_recall")
    pivot = pivot.sort_values("A")
    y = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y - 0.19, pivot["A"], 0.38, label="A : LightGBM + regles",
            color=COLOR_A)
    ax.barh(y + 0.19, pivot["B"], 0.38, label="B : + Isolation Forest",
            color=COLOR_B)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Part des evenements detectes au moins une fois")
    ax.set_xlim(0, 1.08)
    ax.set_title("Rappel au niveau evenement (test du split principal)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def per_class_f1_final(class_summary: pd.DataFrame, path: Path) -> Path:
    """F1 par classe du modele final, avec son ecart-type inter-decoupages."""
    frame = class_summary.sort_values("mean")
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y, frame["mean"], xerr=frame["std"].fillna(0.0), capsize=4,
            color=COLOR_A)
    for i, (mean, std) in enumerate(zip(frame["mean"], frame["std"].fillna(0))):
        ax.text(mean + std + 0.015, i, f"{mean:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(frame.index)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("F1 (moyenne +/- ecart-type sur les decoupages)")
    ax.set_title("F1 par classe — modele final")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, path)


def event_recall_final(events: pd.DataFrame, path: Path) -> Path:
    """Part des evenements detectes au moins une fois, par classe."""
    frame = events.sort_values("event_recall")
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y, frame["event_recall"], color=COLOR_A)
    for i, (value, count) in enumerate(zip(frame["event_recall"],
                                           frame["n_events"])):
        ax.text(value + 0.012, i, f"{value:.2f}  ({int(count)} ev.)",
                va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["label"])
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Part des evenements detectes au moins une fois")
    ax.set_title("Rappel au niveau evenement — modele final")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, path)


# -------------------------------------------------------------------------
# Figures de calibration de confiance
# -------------------------------------------------------------------------

COLOR_RAW, COLOR_CAL = "#C8622B", "#2E6F9E"
PALETTE = ["#7A8B99", "#2E6F9E", "#4C9A6A", "#C8922B"]
WEAK = ["sensor_bias", "sensor_drift"]


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def reliability_figure(reliability: pd.DataFrame, best: str, path: Path) -> Path:
    """Diagramme de fiabilite : confiance annoncee contre exactitude observee.

    La diagonale est la calibration parfaite. Au-dessous, le modele est
    sur-confiant ; au-dessus, sous-confiant.
    """
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="calibration parfaite")
    for method, color, marker in ((("aucune"), COLOR_RAW, "o"),
                                  ((best), COLOR_CAL, "s")):
        sub = (reliability[reliability.method == method]
               .groupby(["bin_low", "bin_high"])
               .apply(lambda g: pd.Series({
                   "confidence": np.average(g.confidence, weights=g.n_rows),
                   "accuracy": np.average(g.accuracy, weights=g.n_rows),
                   "n_rows": g.n_rows.sum()}), include_groups=False)
               .reset_index())
        sub = sub[sub.n_rows >= 50]
        label = "sans calibration" if method == "aucune" else f"calibre ({method})"
        ax.plot(sub.confidence, sub.accuracy, marker + "-", color=color,
                linewidth=2, markersize=7, label=label)
    ax.set_xlabel("Confiance annoncee par le modele")
    ax.set_ylabel("Exactitude reellement observee")
    ax.set_title("Diagramme de fiabilite (test, 5 decoupages)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")
    return _save(fig, path)


def confidence_histogram(store: dict, best_proba: dict, path: Path) -> Path:
    """Repartition des confiances avant et apres calibration."""
    raw = np.concatenate([e["proba_test"].max(axis=1) for e in store.values()])
    cal = np.concatenate([p.max(axis=1) for p in best_proba.values()])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(raw, bins=50, alpha=0.55, color=COLOR_RAW, density=True,
            label=f"sans calibration (moy {raw.mean():.3f})")
    ax.hist(cal, bins=50, alpha=0.55, color=COLOR_CAL, density=True,
            label=f"calibre (moy {cal.mean():.3f})")
    ax.set_xlabel("Confiance de la classe predite")
    ax.set_ylabel("Densite")
    ax.set_title("Distribution des confiances : le modele cesse de surestimer")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def metrics_figure(summary: pd.DataFrame, path: Path) -> Path:
    """ECE, MCE et Brier pour chaque methode de calibration."""
    methods = summary["method"].tolist()
    metrics = [("test_ece", "ECE"), ("test_mce", "MCE"), ("test_brier", "Brier")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (column, title) in zip(axes, metrics):
        values = summary[column].tolist()
        bars = ax.bar(methods, values,
                      color=[PALETTE[i % len(PALETTE)]
                             for i in range(len(methods))])
        best = int(np.argmin(values))
        bars[best].set_edgecolor("black")
        bars[best].set_linewidth(2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                    f"{v:.4f}", ha="center", fontsize=8.5)
        ax.set_title(f"{title} (test, plus bas = mieux)")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Qualite de calibration par methode")
    return _save(fig, path)


def classwise_figure(classwise: pd.DataFrame, best: str, path: Path) -> Path:
    """ECE par classe : les classes rares sont-elles honnetes elles aussi ?"""
    pivot = (classwise[classwise.method.isin(["aucune", best])]
             .groupby(["label", "method"])["classwise_ece"].mean().unstack())
    pivot = pivot.sort_values("aucune")
    y = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.barh(y - 0.19, pivot["aucune"], 0.38, label="sans calibration",
            color=COLOR_RAW)
    ax.barh(y + 0.19, pivot[best], 0.38, label=f"calibre ({best})",
            color=COLOR_CAL)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("ECE un-contre-tous (plus bas = mieux)")
    ax.set_title("Calibration par classe")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, path)


def coverage_figure(curves: pd.DataFrame, best: str, threshold: float,
                    path: Path) -> Path:
    """Compromis couverture / exactitude de la politique d'abstention."""
    curve = (curves[curves.method == best]
             .groupby("threshold")[["coverage", "accuracy_on_covered"]]
             .mean().reset_index())
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(curve.coverage, curve.accuracy_on_covered, "o-", color=COLOR_CAL,
            linewidth=2, markersize=6)
    for _, row in curve.iterrows():
        if row.threshold in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95):
            ax.annotate(f"seuil {row.threshold:.2f}",
                        (row.coverage, row.accuracy_on_covered),
                        textcoords="offset points", xytext=(6, -11), fontsize=8)
    at = curve[np.isclose(curve.threshold, threshold)]
    if len(at):
        ax.scatter(at.coverage, at.accuracy_on_covered, s=160, facecolors="none",
                   edgecolors="black", linewidths=2, zorder=5,
                   label=f"seuil retenu {threshold:.2f}")
        ax.legend(loc="lower left")
    ax.set_xlabel("Couverture : part des lignes effectivement classees")
    ax.set_ylabel("Exactitude sur les lignes classees")
    ax.set_title("Abstention : ce que l'on gagne en renvoyant INCERTAIN")
    ax.grid(alpha=0.25)
    return _save(fig, path)


def weak_class_confidence_figure(store: dict, best_proba: dict,
                                 path: Path) -> Path:
    """Confiance annoncee sur les classes faibles, avant et apres.

    C'est le benefice concret pour l'exploitant : apres calibration, une
    prediction ``sensor_bias`` ne doit plus s'annoncer avec une confiance
    qu'elle ne merite pas.
    """
    rows = []
    for split, entry in store.items():
        classes = np.asarray(entry["classes"])
        y = entry["y_test"]
        for method, proba in (("sans calibration", entry["proba_test"]),
                              ("calibre", best_proba[split])):
            predicted = classes[proba.argmax(axis=1)]
            confidence = proba.max(axis=1)
            for label in WEAK + ["normal"]:
                mask = predicted == label
                if mask.sum() < 20:
                    continue
                rows.append({
                    "label": label, "method": method,
                    "confidence": float(confidence[mask].mean()),
                    "accuracy": float((y[mask] == label).mean())})
    frame = pd.DataFrame(rows).groupby(["label", "method"])[
        ["confidence", "accuracy"]].mean().reset_index()

    labels = [c for c in WEAK + ["normal"] if c in frame.label.unique()]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (method, color) in enumerate((("sans calibration", COLOR_RAW),
                                         ("calibre", COLOR_CAL))):
        sub = frame[frame.method == method].set_index("label")
        conf = [float(sub.loc[c, "confidence"]) for c in labels]
        ax.bar(x + (i - 0.5) * 0.36, conf, 0.36, color=color,
               label=f"confiance annoncee — {method}")
    accuracy = (frame[frame.method == "calibre"].set_index("label")
                .loc[labels, "accuracy"].tolist())
    ax.plot(x, accuracy, "k_", markersize=38, markeredgewidth=2.5,
            label="exactitude reelle")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Probabilite")
    ax.set_ylim(0, 1.05)
    ax.set_title("Confiance annoncee contre exactitude reelle, par classe predite")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def make_all(cfg, store, results, reliability, classwise, curves, best,
             threshold) -> list[Path]:
    from ..models.calibration import build_calibrators

    best_proba = {}
    for split, entry in store.items():
        calibrator = next(c for c in build_calibrators() if c.name == best)
        calibrator.fit(entry["proba_validation"], entry["y_validation"],
                       entry["classes"])
        best_proba[split] = calibrator.transform(entry["proba_test"])

    summary = (results.groupby("method")
               [["test_ece", "test_mce", "test_brier", "val_nll"]]
               .mean().reset_index().sort_values("val_nll"))

    return [
        reliability_figure(reliability, best,
                           cfg.output("figures", "calibration_reliability.png")),
        metrics_figure(summary,
                       cfg.output("figures", "calibration_metrics.png")),
        confidence_histogram(store, best_proba,
                             cfg.output("figures",
                                        "calibration_confidence_histogram.png")),
        classwise_figure(classwise, best,
                         cfg.output("figures", "calibration_per_class.png")),
        coverage_figure(curves, best, threshold,
                        cfg.output("figures", "calibration_coverage.png")),
        weak_class_confidence_figure(
            store, best_proba,
            cfg.output("figures", "calibration_weak_classes.png")),
    ]
