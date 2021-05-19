#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = os.getenv('PROJECT_ROOT')
if PROJECT_ROOT is None:
    PROJECT_ROOT = '../..'
PROJECT_ROOT = Path(PROJECT_ROOT)

DEFAULT_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'heart.csv'


@click.command()
@click.option("--data_path",
              prompt='Data path:\t',
              default=DEFAULT_DATA_PATH)
def make_request(data_path):
    df = pd.read_csv(data_path)
    df = df.drop('target', axis=1)
    request_features = list(df.columns)
    for i in range(10):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in df.iloc[i].tolist()
        ]
        response = requests.get(
            "http://0.0.0.0:8000/predict/",
            json={"data": [request_data], "features": request_features},
        )
        print(response.status_code)
        print(response.json())


if __name__ == '__main__':
    make_request()
