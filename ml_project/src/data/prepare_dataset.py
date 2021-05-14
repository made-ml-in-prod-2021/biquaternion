#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import SplitConfig
from typing import Tuple


def split_dataset(src_df: pd.DataFrame,
                  split_config: SplitConfig) -> Tuple[pd.DataFrame,
                                                      pd.DataFrame,
                                                      pd.DataFrame]:
    train, test = train_test_split(src_df,
                                   test_size=split_config.test_size,
                                   random_state=split_config.random_state)
    train, val = train_test_split(train,
                                  test_size=split_config.validation_size,
                                  random_state=split_config.random_state)
    return train, val, test


def extract_target(src_df: pd.DataFrame,
                   target_name: str) -> Tuple[pd.DataFrame,
                                              pd.Series]:
    features = src_df.loc[:, src_df.columns != target_name]
    target = src_df[target_name]
    return features, target


if __name__ == '__main__':
    pass
