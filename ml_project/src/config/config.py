#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from omegaconf import MISSING

from .model_config import ModelConfig
from .data_config import SplitConfig, DatasetConfig
from .common_config import CommonConfig


@dataclass
class Config:
    model: ModelConfig = MISSING
    split: SplitConfig = MISSING
    dataset: DatasetConfig = MISSING
    common: CommonConfig = MISSING
