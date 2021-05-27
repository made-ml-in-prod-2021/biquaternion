#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
from pathlib import Path
from typing import Union, List, Optional

import pandas as pd
import uvicorn
from sklearn.base import BaseEstimator
from fastapi import FastAPI
from pydantic import BaseModel, conlist
import logging.config

from src.common import elapsed_timer

PROJECT_ROOT = os.getenv('PROJECT_ROOT')
if PROJECT_ROOT is None:
    PROJECT_ROOT = '../..'
PROJECT_ROOT = Path(PROJECT_ROOT)

MODEL_PATH = PROJECT_ROOT / 'models' / 'model.pickle'

# logging.config.fileConfig(PROJECT_ROOT / 'conf/logging.conf')
logger = logging.getLogger('onlineInferenceService.app')


def load_object(src_path: Union[str, Path]) -> BaseEstimator:
    with open(src_path, 'rb') as f:
        return pickle.load(f)


class InputParamsModel(BaseModel):
    data: List[conlist(Union[int, float, None], min_items=13, max_items=13)]
    features: conlist(str, min_items=13, max_items=13)


class HeartDiseaseResponse(BaseModel):
    score: float
    decision: int
    human_readable: str


model: Optional[BaseEstimator] = None


def predict(data: List,
            features: List[str],
            model: BaseEstimator,
            threshold: float = 0.5) -> List[HeartDiseaseResponse]:
    data = pd.DataFrame(data, columns=features)
    with elapsed_timer() as elapsed:
        score = model.predict(data)
        logger.info(f'elapsed {elapsed():.4f} sec')
    decision = 0 if score < threshold else 1
    human_readable = 'negative' if decision == 0 else 'positive'
    return [HeartDiseaseResponse(score=score, decision=decision, human_readable=human_readable)]


app = FastAPI()


@app.get('/')
def main():
    return 'entry point'


@app.get('/health')
def main():
    return model is not None


@app.on_event('startup')
def load_model():
    global model
    model_path = MODEL_PATH
    logger.info('loading model')
    model = load_object(model_path)
    if model is None:
        logger.error('could not load model')
    else:
        logger.info('model loading succeeded')


@app.get('/predict/', response_model=List[HeartDiseaseResponse])
async def call_predict(request: InputParamsModel, threshold=0.5):
    global model
    if model is not None:
        logger.info('call predict')
        return predict(request.data, request.features, model, threshold=threshold)
    else:
        logger.error('model is None')
        return [HeartDiseaseResponse(score=-1.0, decision=-1, human_readable='unknown (error: model is None)')]


if __name__ == '__main__':
    logger.info('start service')
    uvicorn.run("app:app", host='0.0.0.0', port=os.getenv('PORT', 8000))
