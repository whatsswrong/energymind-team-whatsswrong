"""Les fichiers attendus se chargent et la table assemblee est coherente."""

from __future__ import annotations

import pandas as pd

EXPECTED_DAYS = 240
STEPS_PER_DAY = 96
N_HOUSES = 20
N_ZONES = 4


def test_fichiers_charges(raw):
    assert len(raw.houses) == EXPECTED_DAYS * STEPS_PER_DAY * N_HOUSES
    assert len(raw.zones) == EXPECTED_DAYS * STEPS_PER_DAY * N_ZONES
    assert len(raw.transformer) == EXPECTED_DAYS * STEPS_PER_DAY


def test_timestamps_analyses(raw):
    for frame in (raw.houses, raw.zones, raw.transformer):
        assert pd.api.types.is_datetime64_any_dtype(frame["timestamp"])


def test_cle_maison_unique(raw):
    assert not raw.houses.duplicated(["timestamp", "house_id"]).any()


def test_assemblage_sans_multiplication(merged):
    table, _ = merged
    assert len(table) == EXPECTED_DAYS * STEPS_PER_DAY * N_HOUSES
    assert not table.duplicated(["timestamp", "house_id"]).any()
    assert table[["zone_P", "zone_V", "trans_P"]].notna().all().all()


def test_vue_large_alignee(merged):
    import numpy as np
    table, wide = merged
    assert wide.H == N_HOUSES and wide.Z == N_ZONES
    assert np.allclose(wide.flatten(wide.P), table["P"].to_numpy(),
                       equal_nan=True)


def test_aucun_evenement_a_cheval(splits, merged):
    from src.data.splits import assert_no_leakage
    table, _ = merged
    for split in splits.values():
        assert_no_leakage(split, table)
