import logging
import os
import random
from datetime import timedelta, datetime
from pathlib import Path
from typing import Any

from airflow import DAG

from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago

import pandas as pd

logger = logging.getLogger('retrieve_data_DAG')

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

PROJECT_ROOT = os.getenv('PROJECT_ROOT')
if PROJECT_ROOT is None:
    PROJECT_ROOT = '/opt/airflow'
PROJECT_ROOT = Path(PROJECT_ROOT)

DATA_DIR_PATH = PROJECT_ROOT / 'data'

RAW_DATA_PATH = DATA_DIR_PATH / 'raw'
EXTERNAL_DATA_PATH = DATA_DIR_PATH / 'external'
RECEIVED_DATA_FILENAME = '_data.csv'
FINAL_DATA_FILENAME = 'data.csv'

DEFAULT_AMOUNT = 100


def _generate_csv(amount: int,
                  name_date_template: Any,
                  policy='generate', **context):
    output_dir = EXTERNAL_DATA_PATH / str(name_date_template)
    logger.debug(f'making dir "{output_dir}"')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    src_dataset = RAW_DATA_PATH / 'heart.csv'
    if not src_dataset.exists():
        logger.error('src_dataset not found')
    df = pd.read_csv(src_dataset)
    df = pd.DataFrame(columns=df.columns)
    if policy == 'generate':
        logger.debug('generating data')
        for i in range(amount // 5, amount):
            row = {'age': random.randint(29, 77),
                   'sex': random.randint(0, 1),
                   'cp': random.randint(0, 3),
                   'trestbps': random.randint(94, 200),
                   'chol': random.randint(126, 564),
                   'fbs': random.randint(0, 1),
                   'restecg': random.randint(0, 2),
                   'thalach': random.randint(71, 202),
                   'exang': random.randint(0, 1),
                   'oldpeak': random.uniform(0.0, 6.2),
                   'slope': random.randint(0, 2),
                   'ca': random.randint(0, 4),
                   'thal': random.randint(0, 3),
                   'target': random.randint(0, 1)}
            df = df.append(row, ignore_index=True)
    else:
        logger.error('generate policy is only implemented')
    output_path = output_dir / RECEIVED_DATA_FILENAME
    context['task_instance'].xcom_push(key='latest_data_path', value=str(output_path))
    logger.debug(f'saving retrieved data to {output_path}')
    df.to_csv(output_path, float_format='%.3f', index=False)


def _wait_for_file(**context):
    logger.info('waiting')
    latest_data_path = context['task_instance'].xcom_pull(task_ids='generate_csv',
                                                          key='latest_data_path')
    return Path(latest_data_path).exists()


def _merge_data(**context):
    logger.info('merging')
    latest_data_path = context["task_instance"].xcom_pull(task_ids="generate_csv",
                                                          key="latest_data_path")
    dst_data_path = Path(RAW_DATA_PATH) / FINAL_DATA_FILENAME
    src_data_path = dst_data_path
    if not src_data_path.exists():
        logger.info(f'{src_data_path} doesn\'t exists\nUse heart.csv')
        src_data_path = Path(RAW_DATA_PATH) / 'heart.csv'
    src_df = pd.read_csv(src_data_path)
    latest_df = pd.read_csv(latest_data_path)
    src_df = src_df.append(latest_df)
    src_df.to_csv(dst_data_path, index=False)


with DAG(dag_id='retrieve_data',
         default_args=default_args,
         schedule_interval='@daily',
         start_date=days_ago(5)) as dag:
    task_1 = PythonOperator(task_id='generate_csv',
                            python_callable=_generate_csv,
                            op_kwargs={'amount': DEFAULT_AMOUNT,
                                       'name_date_template': '{{ ds }}'})
    wait_1 = PythonSensor(task_id='wait_for_file',
                          python_callable=_wait_for_file)
    task_2 = PythonOperator(task_id='merge_datasets',
                            python_callable=_merge_data)

    task_1 >> wait_1 >> task_2

if __name__ == '__main__':
    _generate_csv(100, f'test_{datetime.now().date()}')
