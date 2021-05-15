#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging.config
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
from src.common import elapsed_timer

logging.config.fileConfig('conf/logging.conf')
logger = logging.getLogger('mlProject.train_model')

PROJECT_ROOT = os.getenv('PROJECT_ROOT')
if PROJECT_ROOT is None:
    PROJECT_ROOT = '../..'
PROJECT_ROOT = Path(PROJECT_ROOT)


def train_model(model: BaseEstimator,
                x: pd.DataFrame,
                target: pd.Series) -> BaseEstimator:
    logger.info('start model training')
    with elapsed_timer() as elapsed:
        model.fit(x, target)
        logger.info(f'finish model training after {elapsed():.4f} sec')
    return model


def val_model(model: BaseEstimator,
              x: pd.DataFrame,
              target: pd.Series,
              score_fn: DictConfig) -> float:
    logger.info('start model validation')
    with elapsed_timer() as elapsed:
        y = model.predict(x)
        logger.info(f'finish model validation after {elapsed():.4f} sec')
    return hydra.utils.call(score_fn, y, target)


def save_report(model: BaseEstimator,
                cfg: Config,
                metrics: dict,
                project_root: Path) -> None:
    logger.info('start report saving')
    output_path = project_root / 'reports' / datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    output_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f'output_path: {str(output_path)}')
    models_path = project_root / 'models'
    logger.debug(f'models_path: {str(models_path)}')
    metrics_filename = output_path / 'metrics.yaml'
    logger.debug(f'metrics_filename: {str(metrics_filename)}')
    with open(metrics_filename, 'w') as f:
        f.write(yaml.dump(metrics))
    output_config_filename = output_path / 'config.yaml'
    logger.debug(f'output_config_filename: {str(output_config_filename)}')
    with open(output_config_filename, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg=cfg))
    output_model_filename = output_path / 'model.pickle'
    logger.debug(f'output_model_filename: {str(output_model_filename)}')
    with open(output_model_filename, 'wb') as f:
        pickle.dump(model, f)
    latest_model_symlink = models_path / 'latest_model.pickle'
    if Path(latest_model_symlink).exists():
        os.remove(Path(latest_model_symlink))
        logger.info('latest model symlink removed')
    Path(latest_model_symlink).symlink_to(output_model_filename)


@hydra.main(config_path='../../conf', config_name='config')
def main(cfg: Config):
    logger.info('start training pipeline')
    logger.debug(f'config: {OmegaConf.to_yaml(cfg)}')
    dataset_path = PROJECT_ROOT / cfg.dataset.raw_dir_path / cfg.dataset.dataset_filename
    logger.debug(f'dataset_path: {str(dataset_path)}')
    logger.info('reading dataset')
    data = pd.read_csv(dataset_path)
    logger.info('splitting dataset')
    train, val, test = split_dataset(src_df=data,
                                     split_config=cfg.split)
    logger.info('instantiating model')
    model = hydra.utils.instantiate(cfg.model)

    logger.info('extract targets for train')
    x, target = extract_target(train, cfg.common.target_feature)
    logger.debug(f'shapes: x: {x.shape}, target: {target.shape}')
    model = train_model(model, x=x, target=target)

    score_fn = cfg.common.score_fn
    logger.info('extract targets for validation')
    x, target = extract_target(val, cfg.common.target_feature)
    logger.debug(f'shapes: x: {x.shape} target: {target.shape}')
    score = val_model(model,
                      x=x,
                      target=target,
                      score_fn=score_fn)
    logger.info(f'got score: {score:.4f} on validation')

    logger.info('extract targets for test')
    x, target = extract_target(test, cfg.common.target_feature)
    logger.debug(f'shapes: x: {x.shape} target: {target.shape}')
    predictions = predict(model=model,
                          src_df=test)
    logger.debug(f'predictions shape: {predictions.shape}')

    save_report(model=model,
                cfg=cfg,
                metrics=classification_report(y_true=target,
                                              y_pred=predictions,
                                              output_dict=True),
                project_root=PROJECT_ROOT)
    logger.info('training pipeline finished')


if __name__ == '__main__':
    main()
