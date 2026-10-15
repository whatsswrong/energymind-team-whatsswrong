"""Assemblage de la table de travail niveau maison + contexte reseau.

Produit une table ``(timestamp, house_id)`` unique, triee par timestamp puis
house_id — exactement l'ordre de ``raw_houses.csv``. Cet ordre est contractuel :
les features sont calculees en representation large ``(T, H)`` puis aplaties
dans ce meme ordre.

Toutes les jointures sont verifiees : pas de multiplication de lignes, unicite
de la cle, association de zone correcte.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loader import RawData

KEY = ["timestamp", "house_id"]


@dataclass
class WideData:
    """Vue large du reseau : une colonne par maison, une ligne par timestamp.

    Toutes les grandeurs temporelles sont calculees sur ces matrices, ce qui
    rend les operations de voisinage et de fenetre glissante simples et sures
    (aucune jointure implicite, aucun melange d'entites).
    """

    timestamps: pd.DatetimeIndex
    house_ids: list[str]
    zone_ids: list[str]
    zone_of_house: np.ndarray        # (H,) index de zone de chaque maison
    V: np.ndarray                    # (T, H)
    I: np.ndarray
    P: np.ndarray
    PF: np.ndarray
    E: np.ndarray
    zone_P: np.ndarray               # (T, Z)
    zone_V: np.ndarray               # (T, Z)
    trans_P: np.ndarray              # (T,)
    day: np.ndarray                  # (T,) jour de simulation, 1-based
    slot: np.ndarray                 # (T,) index du pas dans la journee
    is_weekend: np.ndarray           # (T,) bool

    @property
    def T(self) -> int:
        return len(self.timestamps)

    @property
    def H(self) -> int:
        return len(self.house_ids)

    @property
    def Z(self) -> int:
        return len(self.zone_ids)

    def neighbours(self, house: int) -> np.ndarray:
        """Indices des autres maisons de la meme zone."""
        same = np.flatnonzero(self.zone_of_house == self.zone_of_house[house])
        return same[same != house]

    def flatten(self, wide: np.ndarray) -> np.ndarray:
        """(T, H) -> vecteur aligne sur la table longue (timestamp, house_id)."""
        return wide.ravel()


def build_house_table(raw: RawData) -> pd.DataFrame:
    """Jointure maison + contexte zone + contexte transformateur, verifiee."""
    houses = raw.houses.sort_values(KEY).reset_index(drop=True)
    n_before = len(houses)

    assert not houses.duplicated(KEY).any(), "cle (timestamp, house_id) non unique"

    merged = houses.merge(raw.zones, on=["timestamp", "zone_id"], how="left",
                          validate="many_to_one")
    assert len(merged) == n_before, (
        f"la jointure zone a multiplie les lignes : {len(merged)} != {n_before}")

    merged = merged.merge(raw.transformer, on="timestamp", how="left",
                          validate="many_to_one")
    assert len(merged) == n_before, (
        f"la jointure transformateur a multiplie les lignes : {len(merged)}")

    missing_ctx = merged[["zone_P", "zone_V", "trans_P"]].isna().any(axis=1).sum()
    assert missing_ctx == 0, f"{missing_ctx} lignes sans contexte zone/transfo"

    # l'association maison -> zone doit etre unique et stable dans le temps
    per_house = merged.groupby("house_id")["zone_id"].nunique()
    assert (per_house == 1).all(), "une maison change de zone selon le timestamp"

    merged["day"] = (merged["timestamp"] - merged["timestamp"].min()).dt.days + 1
    return merged


def attach_labels(table: pd.DataFrame, raw: RawData) -> pd.DataFrame:
    """Ajoute la cible supervisee. ``event_id`` sert uniquement a l'evaluation."""
    house_labels = raw.house_labels[
        ["timestamp", "house_id", "label", "event_id", "severity"]]
    n_before = len(table)
    out = table.merge(house_labels, on=KEY, how="left", validate="one_to_one")
    assert len(out) == n_before, "la jointure des labels a multiplie les lignes"
    assert out["label"].notna().all(), "des observations n'ont pas de label"
    return out


def to_wide(table: pd.DataFrame, raw: RawData) -> WideData:
    """Passe la table longue en matrices (T, H) / (T, Z) / (T,)."""
    table = table.sort_values(KEY)
    timestamps = pd.DatetimeIndex(np.sort(table["timestamp"].unique()))
    house_ids = sorted(table["house_id"].unique())
    zone_ids = sorted(raw.zones["zone_id"].unique())
    T, H = len(timestamps), len(house_ids)
    assert len(table) == T * H, (
        f"table incomplete : {len(table)} lignes pour {T} x {H} attendues")

    mapping = (table.drop_duplicates("house_id").set_index("house_id")["zone_id"])
    zone_of_house = np.array([zone_ids.index(mapping[h]) for h in house_ids])

    wide = {c: table[c].to_numpy(dtype=float).reshape(T, H)
            for c in ("V", "I", "P", "PF", "E")}

    zones = raw.zones.sort_values(["timestamp", "zone_id"])
    zone_P = zones["zone_P"].to_numpy(float).reshape(T, len(zone_ids))
    zone_V = zones["zone_V"].to_numpy(float).reshape(T, len(zone_ids))
    trans = raw.transformer.sort_values("timestamp")["trans_P"].to_numpy(float)

    day = np.asarray((timestamps - timestamps.min()).days) + 1
    minutes = np.asarray(timestamps.hour) * 60 + np.asarray(timestamps.minute)
    deltas = np.diff(timestamps.to_numpy()).astype("timedelta64[m]").astype(int)
    step = int(np.median(deltas)) if deltas.size else 15
    slot = minutes // max(step, 1)
    is_weekend = np.asarray(timestamps.dayofweek) >= 5

    return WideData(timestamps=timestamps, house_ids=house_ids, zone_ids=zone_ids,
                    zone_of_house=zone_of_house, zone_P=zone_P, zone_V=zone_V,
                    trans_P=trans, day=day, slot=slot, is_weekend=is_weekend,
                    **wide)


def merge_network_data(raw: RawData) -> tuple[pd.DataFrame, WideData]:
    """Table longue labellisee + vue large, dans le meme ordre de lignes."""
    table = attach_labels(build_house_table(raw), raw)
    wide = to_wide(table, raw)
    return table.reset_index(drop=True), wide
