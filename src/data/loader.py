"""Chargement et inspection des fichiers CSV produits par le simulateur.

Regle de non-fuite appliquee des le chargement : seuls ``raw_houses.csv``,
``raw_zones.csv`` et ``raw_transformer.csv`` alimentent les features.
``labels.csv`` est charge separement et ne sert que de cible supervisee et
d'outillage d'evaluation (``event_id`` compris).

``ground_truth.csv``, ``injection_log.csv`` et ``house_profiles.csv`` ne sont
jamais lus par ce module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_FILES = {
    "houses": ("raw_houses.csv", ["timestamp", "house_id", "zone_id",
                                  "V", "I", "P", "PF", "E"]),
    "zones": ("raw_zones.csv", ["timestamp", "zone_id", "zone_P", "zone_V"]),
    "transformer": ("raw_transformer.csv", ["timestamp", "trans_P"]),
}

#: cle d'unicite de chaque fichier de mesures. Un doublon fausserait les
#: jointures et les remises en forme (T, H) sans erreur visible : on refuse
#: donc le fichier des le chargement.
UNIQUE_KEYS = {
    "houses": ["timestamp", "house_id"],
    "zones": ["timestamp", "zone_id"],
    "transformer": ["timestamp"],
}
LABEL_FILE = "labels.csv"

#: colonnes de labels.csv formellement interdites comme features
FORBIDDEN_AS_FEATURES = ("label", "event_id", "anomaly_family", "severity",
                         "target_type", "is_anomaly", "entity_type")


@dataclass
class RawData:
    houses: pd.DataFrame
    zones: pd.DataFrame
    transformer: pd.DataFrame
    labels: pd.DataFrame

    @property
    def house_labels(self) -> pd.DataFrame:
        return self.labels[self.labels["entity_type"] == "house"]

    @property
    def zone_labels(self) -> pd.DataFrame:
        return self.labels[self.labels["entity_type"] == "zone"]

    @property
    def transformer_labels(self) -> pd.DataFrame:
        return self.labels[self.labels["entity_type"] == "transformer"]


def _assert_unique(df: pd.DataFrame, keys: list[str], entity: str) -> None:
    """Refuse tout fichier comportant des cles de telemetrie dupliquees."""
    duplicates = int(df.duplicated(keys).sum())
    if duplicates > 0:
        examples = (df.loc[df.duplicated(keys, keep=False), keys]
                    .drop_duplicates().head(3).to_dict("records"))
        raise ValueError(
            f"{entity} : {duplicates} enregistrement(s) duplique(s) sur la cle "
            f"{keys}. Exemples : {examples}")


def _read(path: Path, expected_columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    df = pd.read_csv(path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if expected_columns is not None:
        missing = [c for c in expected_columns if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} : colonnes absentes {missing} "
                             f"(trouvees : {list(df.columns)})")
        extra = [c for c in df.columns if c not in expected_columns]
        if extra:
            raise ValueError(f"{path.name} : colonnes inattendues {extra} — "
                             "verifier qu'aucune verite de simulation n'a fuite")
    return df


def load_raw_data(data_dir: Path) -> RawData:
    """Charge les trois fichiers de mesures + les labels, schemas verifies."""
    frames = {}
    for name, (fname, cols) in RAW_FILES.items():
        frame = _read(data_dir / fname, cols)
        _assert_unique(frame, UNIQUE_KEYS[name], fname)
        frames[name] = frame
    labels = _read(data_dir / LABEL_FILE)
    _assert_unique(labels, ["timestamp", "entity_type", "entity_id"],
                   LABEL_FILE)
    return RawData(houses=frames["houses"], zones=frames["zones"],
                   transformer=frames["transformer"], labels=labels)


def inspect_files(data: RawData) -> pd.DataFrame:
    """Rapport d'inspection : une ligne par fichier."""
    rows = []
    for name, df, keys in (
        ("raw_houses", data.houses, ["timestamp", "house_id"]),
        ("raw_zones", data.zones, ["timestamp", "zone_id"]),
        ("raw_transformer", data.transformer, ["timestamp"]),
        ("labels", data.labels, ["timestamp", "entity_type", "entity_id"]),
    ):
        deltas = np.diff(np.sort(df["timestamp"].unique())).astype(
            "timedelta64[m]").astype(int)
        rows.append({
            "file": name,
            "rows": len(df),
            "columns": len(df.columns),
            "ts_min": df["timestamp"].min(),
            "ts_max": df["timestamp"].max(),
            "ts_unique": df["timestamp"].nunique(),
            "step_minutes": int(np.median(deltas)) if deltas.size else 0,
            "duplicate_keys": int(df.duplicated(keys).sum()),
            "missing_cells": int(df.isna().sum().sum()),
            "missing_columns": ", ".join(
                f"{k}={int(v)}" for k, v in df.isna().sum().items() if v) or "-",
        })
    return pd.DataFrame(rows)


def expected_shapes(data: RawData) -> pd.DataFrame:
    """Compare les tailles reelles aux tailles attendues (60 j x 96 pas)."""
    n_ts = data.houses["timestamp"].nunique()
    n_houses = data.houses["house_id"].nunique()
    n_zones = data.zones["zone_id"].nunique()
    checks = [
        ("raw_houses", len(data.houses), n_ts * n_houses),
        ("raw_zones", len(data.zones), n_ts * n_zones),
        ("raw_transformer", len(data.transformer), n_ts),
        ("labels", len(data.labels), n_ts * (n_houses + n_zones + 1)),
    ]
    return pd.DataFrame(
        [{"file": f, "rows": r, "expected": e, "match": r == e}
         for f, r, e in checks])


def label_summary(data: RawData) -> pd.DataFrame:
    """Distribution des labels et nombre d'EVENEMENTS par classe et entite."""
    rows = []
    for entity, df in (("house", data.house_labels),
                       ("zone", data.zone_labels),
                       ("transformer", data.transformer_labels)):
        total = len(df)
        for label, sub in df.groupby("label"):
            rows.append({
                "entity_type": entity,
                "label": label,
                "rows": len(sub),
                "pct_rows": round(100.0 * len(sub) / total, 3),
                "events": int(sub["event_id"].nunique()) if label != "normal" else 0,
            })
    return pd.DataFrame(rows).sort_values(["entity_type", "rows"],
                                          ascending=[True, False])
