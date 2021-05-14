#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.base import BaseEstimator

from src.config import Config
from src.data import split_dataset, extract_target


def train_model(model: BaseEstimator,
                x: pd.DataFrame,
                target: pd.Series) -> BaseEstimator:
    model.fit(x, target)
    return model


def val_model(model: BaseEstimator,
              x: pd.DataFrame,
              target: pd.Series,
              score_fn: DictConfig) -> float:
    y = model.predict(x)
    return hydra.utils.call(score_fn, *(y, target))


@hydra.main(config_path='../../conf', config_name='config')
def main(cfg: Config):
    project_root = os.getenv('PROJECT_ROOT')
    if project_root is None:
        project_root = '../..'
    project_root = Path(project_root)
    dataset_path = project_root / \
                   cfg.dataset.raw_dir_path / \
                   cfg.dataset.dataset_filename
    data = pd.read_csv(dataset_path)
    train, val, test = split_dataset(src_df=data,
                                     split_config=cfg.split)

    model = hydra.utils.instantiate(cfg.model)

    x, target = extract_target(train, cfg.common.target_feature)
    model = train_model(model, x=x, target=target)
    score_fn = cfg.common.score_fn
    x, target = extract_target(val, cfg.common.target_feature)
    score = val_model(model,
                      x=x,
                      target=target,
                      score_fn=score_fn)
    print(score)
    x, target = extract_target(test, cfg.common.target_feature)
    score = val_model(model,
                      x=x,
                      target=target,
                      score_fn=score_fn)
    print(score)


if __name__ == '__main__':
    main()
