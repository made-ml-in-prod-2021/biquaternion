#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import pandas as pd
import pytest
from hydra.experimental import initialize, compose

from src.config import Config
from src.data import split_dataset, extract_target

PROJECT_ROOT = os.getenv('PROJECT_ROOT')
if PROJECT_ROOT is None:
    PROJECT_ROOT = Path('.')
PROJECT_ROOT = Path(PROJECT_ROOT)


@pytest.fixture
def get_config() -> Config:
    # conf_path = os.path.relpath(PROJECT_ROOT / 'conf', os.getcwd())
    conf_path = '../../conf'
    with initialize(config_path=str(conf_path)):
        cfg = compose(config_name='config')
    return cfg


def test_prepare_dataset(get_config):
    data_path = PROJECT_ROOT / 'data' / 'raw' / 'heart.csv'
    assert data_path.exists()
    df = pd.read_csv(data_path)
    assert len(df.columns) > 0
    assert 'target' in df.keys()
    cfg = get_config
    assert cfg is not None
    train_df, val_df, _ = split_dataset(df, split_config=cfg.split)
    assert len(train_df) > 0
    assert len(val_df) > 0
    x, t = extract_target(df, target_name='target')
    assert isinstance(x, pd.DataFrame)
    assert isinstance(t, pd.Series)
    assert len(x) == len(t)


if __name__ == '__main__':
    pass
