#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle

import click
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator

from src.data import extract_target

PROJECT_ROOT = os.getenv('PROJECT_ROOT')
if PROJECT_ROOT is None:
    PROJECT_ROOT = Path('../..')
PROJECT_ROOT = Path(PROJECT_ROOT)
DEFAULT_DATA_PATH = PROJECT_ROOT / 'data/raw/heart.csv'
DEFAULT_MODEL_PATH = PROJECT_ROOT / 'models/model.pickle'
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / 'output/predictions.csv'


def predict(model: BaseEstimator,
            src_df: pd.DataFrame) -> pd.Series:
    x, target = extract_target(src_df, 'target')
    return pd.Series(model.predict(x),
                     index=src_df.index,
                     name='prediction')


@click.command()
@click.option("--data_path",
              prompt='Data path:\t',
              default=DEFAULT_DATA_PATH)
@click.option("--model_path",
              prompt='Model path:\t',
              default=DEFAULT_MODEL_PATH)
@click.option("--output_path",
              prompt='Output path:\t',
              default=DEFAULT_OUTPUT_PATH)
def main(data_path, model_path, output_path):
    df = pd.read_csv(data_path)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    predictions = predict(model, df)
    predictions.to_csv(output_path)


if __name__ == '__main__':
    main()
