#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List

from omegaconf import MISSING


@dataclass
class ModelConfig:
    _target_: str = MISSING


@dataclass
class LogisticRegressionConfig(ModelConfig):
    tol: float = MISSING
    C: float = MISSING
    random_state: int = MISSING
    max_iter: int = MISSING


@dataclass
class PerceptronConfig(ModelConfig):
    hidden_layer_sizes: List = MISSING
    activation: str = MISSING
    alpha: float = MISSING
    batch_size: int = MISSING
    learning_rate_init: float = MISSING
    max_iter: int = MISSING
    shuffle: bool = MISSING
    random_state: int = MISSING
    tol: float = MISSING
    momentum: float = MISSING
    early_stopping: bool = MISSING
    n_iter_no_change: int = MISSING


@dataclass
class LinearSvcConfig(ModelConfig):
    tol: float = MISSING
    C: float = MISSING
    random_state: int = MISSING
    max_iter: int = MISSING


@dataclass
class DecisionTreeClassifierConfig(ModelConfig):
    max_depth: int = MISSING
    min_samples_split: int = MISSING
    random_state: int = MISSING
