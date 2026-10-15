"""Fixtures partagees : les donnees ne sont chargees qu'une fois par session."""

from __future__ import annotations

import pytest

from src.config import load_config
from src.data.loader import load_raw_data
from src.data.merge import merge_network_data
from src.data.splits import build_all_splits


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw(cfg):
    return load_raw_data(cfg.data_dir)


@pytest.fixture(scope="session")
def merged(raw):
    return merge_network_data(raw)


@pytest.fixture(scope="session")
def splits(cfg, merged):
    table, wide = merged
    return build_all_splits(cfg, table, wide.timestamps)
