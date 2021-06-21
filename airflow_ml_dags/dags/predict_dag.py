import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger('predict_DAG')

PROJECT_ROOT = os.getenv('PROJECT_ROOT')
if PROJECT_ROOT is None:
    PROJECT_ROOT = '/opt/airflow'
PROJECT_ROOT = Path(PROJECT_ROOT)

DATA_DIR_PATH = PROJECT_ROOT / 'data'

RAW_DATA_PATH = DATA_DIR_PATH / 'raw'
DATA_FILENAME = 'data.csv'

MODELS_DIR_PATH = PROJECT_ROOT / 'models'
MODEL_FILENAME = 'model.dat'

PREDICTIONS_DIR_PATH = DATA_DIR_PATH / 'predictions'
PREDICTIONS_FILENAME = 'predictions.csv'


def _predict(name_date_template):
    logger.info('load data')
    data_path = Path(RAW_DATA_PATH) / DATA_FILENAME
    df = pd.read_csv(data_path)
    logger.info('load model')
    model_path = Variable.get('model', str(MODELS_DIR_PATH / 'latest' / MODEL_FILENAME))
    if not Path(model_path).exists():
        logger.info('model doesn\'t exist')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    x_val, y_val = df.drop('target', axis=1), df['target']
    y_pred = model.predict(x_val)
    y_pred = pd.DataFrame(y_pred)
    output_dir = PREDICTIONS_DIR_PATH / name_date_template
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / PREDICTIONS_FILENAME
    y_pred.to_csv(output_path, index=False)


with DAG(dag_id='predict_dag',
         schedule_interval='@daily',
         start_date=days_ago(5)) as dag:
    predict = PythonOperator(task_id='predict',
                             python_callable=_predict,
                             op_kwargs={'name_date_template': '{{ ds }}'})
    end_task = DummyOperator(task_id='end')

    predict >> end_task
