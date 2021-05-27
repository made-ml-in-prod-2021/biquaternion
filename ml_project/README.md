# Домашняя работа "Распознавание заболеваний сердца"

## Постановка задачи

Разработать программное обеспечение для распознавания заболеваний сердца на основе машинного обучения 

## Поддерживаемые ОС

* GNU/Linux

## Данные

Использованы данные из следующих источников:  
* https://www.kaggle.com/ronitf/heart-disease-uci

## Установка

для установки необходимо сменить директорию на ml_project, и выполнить следующие команды:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Перед началом работы
нужно выполнить скрипт `setenv.sh`
следующей командой

    source setenv.sh

В результате установится переменная PROJECT_ROOT, которая 
используется в дальнейшем, и создадутся папки 
`logs`, `reports` и `output`

## Подготовка датасета
Необходимо скачать датасет с помощью следующих команды

    cd src/data
    ./retrieve_data
    cd -

## Конфигурация

Для настройки проекта используется [hydra](https://hydra.cc/). 
Все конфигурационные файлы находятся в директории conf.
При добавлении новых параметров конфигурации необходимо также внести изменения в соответствующих файлах src/config/*.py.

## Запуск тренировки

Производится командой

    python3 src/models/train_model.py

В результате в директории reports появится новая директория с датой 
и временем последнего запуска тренировки. Она содержит файлы 
`config.yaml`, `metrics.yaml` и `model.pickle`

При этом в директории ml_project/models появляется
символьная ссылка `latest_model.pickle` на последнюю версию модели

## Предсказания
Выполняются командой

    python3 src/models/predict.py
Ей можно передать входные данные ключами 
`--data_path`, `--model_path` и `--output_path` или ввести эти данные 
в интерактивном режиме, просто не указав ключ(и)  
В результате в директории, указанной как `output_path` будут предсказания 
модели, расположенной в `model_path`, на данных, указанных в `data_path`

## Тестирование

    pytest

## Структура проекта
    .
    ├── conf
    │   ├── common
    │   │   └── common_config.yaml
    │   ├── dataset
    │   │   └── dataset.yaml                
    │   ├── model
    │   │   ├── decision_tree.yaml
    │   │   ├── linear_svc.yaml
    │   │   ├── logistic_regression.yaml
    │   │   └── perceptron.yaml
    │   ├── split
    │   │   └── split.yaml
    │   └── config.yaml
    │   ├── logging.conf
    ├── data
    │    ├── external
    │    │    └── heart-disease-uci.zip
    │    └── raw
    │        └── heart.csv
    ├── logs
    │    └── debug.log
    │    └── error.log
    ├── models
    │    └── latest_model.pickle -> symlink to latest model.pickle
    ├── notebooks
    │    └── EDA.ipynb
    ├── output
    │    └── predictions.csv
    ├── reports
    │    └── 16_05_2021_21_52_38
    │        ├── config.yaml
    │        ├── metrics.yaml
    │        └── model.pickle
    ├── requirements.txt
    ├── setenv.sh
    ├── src
    │    ├── common
    │    │    ├── __init__.py
    │    │    └── timing.py
    │    ├── config
    │    │    ├── common_config.py
    │    │    ├── config.py
    │    │    ├── data_config.py
    │    │    ├── __init__.py
    │    │    └── model_config.py
    │    ├── data
    │    │    ├── __init__.py
    │    │    ├── prepare_dataset.py
    │    │    └── retrieve_data.sh
    │    └── models
    │        ├── predict.py
    │        └── train_model.py
    └── tests
        ├── data
        │    ├── __init__.py
        │    └── test_dataset.py
        ├── __init__.py
        └── models
            ├── __init__.py
            └── test_models.py
