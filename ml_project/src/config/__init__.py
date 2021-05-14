#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from hydra.core.config_store import ConfigStore

from .config import Config
from .data_config import SplitConfig, DatasetConfig
from .model_config import LogisticRegressionConfig, PerceptronConfig, \
    LinearSvcConfig, DecisionTreeClassifierConfig

__all__ = ['Config',
           'SplitConfig',
           'DatasetConfig']

cs = ConfigStore.instance()
cs.store(name='config', node=Config)
cs.store(name='split', node=SplitConfig)
cs.store(name='dataset', node=DatasetConfig)
cs.store(group='model', name='logistic_regression', node=LogisticRegressionConfig)
cs.store(group='model', name='perceptron', node=PerceptronConfig)
cs.store(group='model', name='linear_svc', node=LinearSvcConfig)
cs.store(group='model', name='decision_tree', node=DecisionTreeClassifierConfig)
