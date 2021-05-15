#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import os
import pickle
from pathlib import Path
import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report

from src.config import Config
from src.data import split_dataset, extract_target
from src.models.predict import predict


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
    return hydra.utils.call(score_fn, y, target)


def save_report(model: BaseEstimator,
                cfg: Config,
                metrics: dict,
                project_root: Path) -> None:
    output_path = project_root / 'reports' / datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    output_path.mkdir(parents=True, exist_ok=True)
    models_path = project_root / 'models'
    with open(output_path / 'metrics.yaml', 'w') as f:
        f.write(yaml.dump(metrics))
    with open(output_path / 'config.yaml', 'w') as f:
        f.write(OmegaConf.to_yaml(cfg=cfg))
    with open(output_path / 'model.pickle', 'wb') as f:
        pickle.dump(model, f)
    Path(models_path / 'latest_model.pickle').symlink_to(output_path / 'model.pickle')


@hydra.main(config_path='../../conf', config_name='config')
def main(cfg: Config):
    project_root = os.getenv('PROJECT_ROOT')
    if project_root is None:
        project_root = '../..'
    project_root = Path(project_root)
    dataset_path = project_root / cfg.dataset.raw_dir_path / cfg.dataset.dataset_filename
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
    predictions = predict(model=model,
                          src_df=test)

    save_report(model=model,
                cfg=cfg,
                metrics=classification_report(y_true=target,
                                              y_pred=predictions),
                project_root=project_root)


if __name__ == '__main__':
    main()
