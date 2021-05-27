#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from app import app
from src.requests.make_request import make_request_params

PROJECT_ROOT = os.getenv('PROJECT_ROOT')
if PROJECT_ROOT is None:
    PROJECT_ROOT = '../..'
PROJECT_ROOT = Path(PROJECT_ROOT)


@pytest.fixture
def data_path() -> str:
    return str(PROJECT_ROOT / 'data' / 'raw' / 'heart.csv')


@pytest.fixture
def request_params(data_path) -> dict:
    return next(make_request_params(data_path=data_path))


@pytest.fixture
def expected_prediction() -> tuple:
    return 1, 'positive'


def test_get_entry_point():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == 'entry point'


def test_get_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.content == b'true'


def test_get_predict(request_params, expected_prediction):
    with TestClient(app) as client:
        response = client.get("/predict", json=request_params)
        assert response.status_code == 200
        response_json = response.json()
        assert isinstance(response_json, list)
        response_json = response_json[0]
        assert isinstance(response_json, dict)
        assert response_json['score'] >= 0.0
        assert response_json['score'] <= 1.0
        assert response_json['decision'] in [1, 0]
        assert response_json['human_readable'] in ['negative', 'positive']
        expected_decision, expected_human_readable = expected_prediction
        assert response_json['decision'] == expected_decision
        assert response_json['human_readable'] == expected_human_readable


if __name__ == '__main__':
    pass
