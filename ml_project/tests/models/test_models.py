#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
from pathlib import Path
from time import sleep
from typing import Union

import pandas as pd
import pytest
import sklearn.base
from hydra.experimental import initialize, compose
from tqdm import tqdm

from src.config import Config
from src.models.predict import predict
from src.models.train_model import train_pipeline

PROJECT_ROOT = os.getenv('PROJECT_ROOT')
if PROJECT_ROOT is None:
    PROJECT_ROOT = Path('.')
PROJECT_ROOT = Path(PROJECT_ROOT)


@pytest.fixture
def tests_sandbox_path() -> Path:
    return Path('sandbox')


@pytest.fixture
def config(tests_sandbox_path) -> Config:
    # conf_path = os.path.relpath(PROJECT_ROOT / 'conf', os.getcwd())
    conf_path = '../../conf'
    with initialize(config_path=str(conf_path)):
        cfg: Config = compose(config_name='config')
    cfg.common.reports_path = str(Path(tests_sandbox_path) / cfg.common.reports_path)
    cfg.common.models_path = str(Path(tests_sandbox_path) / cfg.common.models_path)
    return cfg


@pytest.fixture
def dataset(path: Union[Path, str]) -> pd.DataFrame:
    return pd.read_csv(path)


@pytest.fixture
def model(config: Config):
    train_pipeline(config)
    model_path = Path(config.common.models_path) / 'latest_model.pickle'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def test_models(model, dataset, config: Config):
    assert (issubclass(type(model), sklearn.base.BaseEstimator))
    for i in tqdm(range(10)):
        train_pipeline(config)
        # sleep(1)
    assert Path(config.common.reports_path).exists()
    assert (Path(config.common.models_path) / 'latest_model.pickle').exists()
    for metrics in Path(config.common.reports_path).glob('*/.metrics.yams'):
        assert metrics['accuracy'] > 0.0
    preds = predict(model, dataset)
    assert isinstance(preds, pd.Series)



if __name__ == '__main__':
    pass
