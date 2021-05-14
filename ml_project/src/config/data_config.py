#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class SplitConfig:
    name: str = MISSING
    validation_size: float = MISSING
    test_size: float = MISSING
    random_state: int = MISSING


@dataclass
class DatasetConfig:
    raw_dir_path: str = MISSING
    processed_dir_path: str = MISSING
    dataset_filename: str = MISSING
