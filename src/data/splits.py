"""Decoupages temporels, et protection contre la fuite au niveau EVENEMENT.

Les observations sont temporelles : un melange aleatoire des lignes mettrait
des pas voisins du meme evenement de part et d'autre de la frontiere. On
decoupe donc par jour de simulation.

Protection supplementaire : si un evenement chevauche une frontiere du
decoupage, TOUTES ses lignes sont ecartees de l'experience (pour ce split
uniquement). Sans cela, un meme evenement apparaitrait a la fois en
entrainement et en test, ce qui gonflerait artificiellement les scores.

``event_id`` n'est utilise qu'ici et dans l'evaluation : il n'entre jamais dans
la matrice de features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PARTS = ("train", "validation", "test")


@dataclass
class Split:
    name: str
    days: dict[str, tuple[int, int]]
    rows: dict[str, np.ndarray]        # masques booleens sur la table longue
    steps: dict[str, np.ndarray]       # indices de pas de temps (vue large)
    dropped_events: list[str]
    dropped_rows: int

    def part_rows(self, part: str) -> np.ndarray:
        return self.rows[part]

    def describe(self) -> str:
        spans = ", ".join(f"{p}=j{self.days[p][0]}-{self.days[p][1]}"
                          for p in PARTS)
        return (f"{self.name} ({spans}) ; {self.dropped_rows} ligne(s) ecartees "
                f"pour {len(self.dropped_events)} evenement(s) a cheval")


def _crossing_events(table: pd.DataFrame, days: dict) -> list[str]:
    """Evenements dont les lignes tombent dans plusieurs parties du decoupage."""
    part_of_day = {}
    for part, (lo, hi) in days.items():
        for d in range(int(lo), int(hi) + 1):
            part_of_day[d] = part
    events = table.loc[table["event_id"].notna(), ["event_id", "day"]].copy()
    if events.empty:
        return []
    events["part"] = events["day"].map(part_of_day)
    per_event = events.groupby("event_id")["part"].nunique(dropna=False)
    spans = per_event[per_event > 1].index.tolist()
    # un evenement partiellement hors decoupage compte aussi comme a cheval
    partial = events.groupby("event_id")["part"].apply(lambda s: s.isna().any())
    return sorted(set(spans) | set(partial[partial].index.tolist()))


def build_split(name: str, days: dict, table: pd.DataFrame,
                timestamps: pd.DatetimeIndex) -> Split:
    """Construit un decoupage et neutralise les evenements a cheval."""
    days = {p: (int(v[0]), int(v[1])) for p, v in days.items()}
    for part in PARTS:
        assert part in days, f"decoupage {name} : partie {part} absente"
    bounds = [days[p] for p in PARTS]
    for (lo1, hi1), (lo2, hi2) in zip(bounds, bounds[1:]):
        assert hi1 < lo2, (f"decoupage {name} : les periodes se chevauchent "
                           f"({hi1} >= {lo2})")

    crossing = _crossing_events(table, days)
    bad = table["event_id"].isin(crossing) if crossing else pd.Series(
        False, index=table.index)

    day_col = table["day"].to_numpy()
    step_days = np.asarray((timestamps - timestamps.min()).days) + 1

    rows, steps = {}, {}
    for part in PARTS:
        lo, hi = days[part]
        in_part = (day_col >= lo) & (day_col <= hi)
        rows[part] = in_part & (~bad.to_numpy())
        steps[part] = np.flatnonzero((step_days >= lo) & (step_days <= hi))
        assert rows[part].sum() > 0, f"{name}/{part} : aucune ligne"

    return Split(name=name, days=days, rows=rows, steps=steps,
                 dropped_events=crossing, dropped_rows=int(bad.sum()))


def build_all_splits(cfg, table: pd.DataFrame,
                     timestamps: pd.DatetimeIndex) -> dict[str, Split]:
    """Split principal + splits glissants, dans l'ordre de la configuration."""
    out = {}
    primary = cfg.section("splits.primary")
    out[primary["name"]] = build_split(
        primary["name"], {p: primary[p] for p in PARTS}, table, timestamps)
    for spec in cfg.get("splits.rolling"):
        out[spec["name"]] = build_split(
            spec["name"], {p: spec[p] for p in PARTS}, table, timestamps)
    return out


def split_class_report(split: Split, table: pd.DataFrame) -> pd.DataFrame:
    """Lignes et EVENEMENTS independants par classe et par partie."""
    rows = []
    for part in PARTS:
        sub = table.loc[split.rows[part]]
        for label, grp in sub.groupby("label"):
            rows.append({
                "split": split.name, "part": part, "label": label,
                "rows": len(grp),
                "events": int(grp["event_id"].nunique()) if label != "normal" else 0,
            })
    return pd.DataFrame(rows)


def assert_no_leakage(split: Split, table: pd.DataFrame) -> None:
    """Aucun evenement ne doit apparaitre dans deux parties du decoupage."""
    seen: dict[str, str] = {}
    for part in PARTS:
        ids = table.loc[split.rows[part], "event_id"].dropna().unique()
        for eid in ids:
            if eid in seen and seen[eid] != part:
                raise AssertionError(
                    f"{split.name} : l'evenement {eid} apparait dans "
                    f"{seen[eid]} et {part}")
            seen[eid] = part
    overlap = (split.rows["train"] & split.rows["test"]).sum()
    assert overlap == 0, f"{split.name} : {overlap} lignes communes train/test"
