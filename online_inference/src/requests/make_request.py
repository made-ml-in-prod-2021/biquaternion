#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Union

import click
import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = os.getenv('PROJECT_ROOT')
if PROJECT_ROOT is None:
    PROJECT_ROOT = '../..'
PROJECT_ROOT = Path(PROJECT_ROOT)

DEFAULT_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'heart.csv'


def make_request_params(data_path: Union[str, Path]) -> dict:
    df = pd.read_csv(data_path)
    df = df.drop('target', axis=1)
    request_features = list(df.columns)
    for row in df.iterrows():
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in list(row[1])
        ]
        yield {'data': [request_data], 'features': request_features}


@click.command()
@click.option("--data_path",
              prompt='Data path:\t',
              default=DEFAULT_DATA_PATH)
def make_request(data_path: Union[str, Path]):
    for i in range(10):
        params = next(make_request_params(data_path=data_path))
        response = requests.get(
            "http://0.0.0.0:8000/predict/",
            json=params
        )
        print(response.status_code)
        if not str(response.status_code).startswith('5'):
            print(response.json())


if __name__ == '__main__':
    make_request()
