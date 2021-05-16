#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any
from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class CommonConfig:
    random_seed: int = MISSING
    target_feature: str = MISSING
    reports_path: str = MISSING
    models_path: str = MISSING
    score_fn: Any = MISSING
