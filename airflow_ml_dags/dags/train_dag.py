import logging
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger('train_DAG')

PROJECT_ROOT = os.getenv('PROJECT_ROOT')
if PROJECT_ROOT is None:
    PROJECT_ROOT = '/opt/airflow'
PROJECT_ROOT = Path(PROJECT_ROOT)

DATA_DIR_PATH = PROJECT_ROOT / 'data'

RAW_DATA_PATH = DATA_DIR_PATH / 'raw'
PROCESSED_DATA_PATH = DATA_DIR_PATH / 'processed'

DATA_FILENAME = 'data.csv'
TRAIN_DATA_FILENAME = 'train.csv'
VAL_DATA_FILENAME = 'val.csv'

MODELS_DIR_PATH = PROJECT_ROOT / 'models'
MODEL_FILENAME = 'model.dat'

TRAIN_VAL_RATIO = 4  # train : val    4 : 1


def _wait_0():
    logger.info('waiting for source dataset')
    return (Path(RAW_DATA_PATH) / DATA_FILENAME).exists()


def _preprocess():
    src_filename = Path(RAW_DATA_PATH) / DATA_FILENAME
    dst_filename = Path(PROCESSED_DATA_PATH) / DATA_FILENAME
    logger.info(f'preprocessing: copying from "{src_filename}"\tto "{dst_filename}"')
    shutil.copy(src_filename, dst_filename)


def _wait_1():
    logger.info('waiting for preprocessed dataset')
    return (Path(PROCESSED_DATA_PATH) / DATA_FILENAME).exists()


def _split():
    path_to_split = Path(PROCESSED_DATA_PATH) / DATA_FILENAME
    train_path = Path(PROCESSED_DATA_PATH) / TRAIN_DATA_FILENAME
    val_path = Path(PROCESSED_DATA_PATH) / VAL_DATA_FILENAME
    logger.info(f'split\t"{path_to_split}"\tinto\t"{train_path}"\tand\t"{val_path}"')
    src_df = pd.read_csv(path_to_split)
    mask = np.random.rand(len(src_df)) < 1 - (1 / (TRAIN_VAL_RATIO + 1))
    train_df = src_df[mask]
    val_df = src_df[~mask]
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)


def _wait_2():
    logger.info('waiting for split')
    return (Path(PROCESSED_DATA_PATH) / TRAIN_DATA_FILENAME).exists() and \
           (Path(PROCESSED_DATA_PATH) / VAL_DATA_FILENAME).exists()


def _train(name_date_template, **context):
    train_path = Path(PROCESSED_DATA_PATH) / TRAIN_DATA_FILENAME
    train_df = pd.read_csv(train_path)
    logger.info('extract targets')
    x_train, y_train = train_df.drop('target', axis=1), train_df['target']
    logger.info('fit model')
    model = DecisionTreeClassifier(max_depth=5,
                                   min_samples_split=2,
                                   random_state=100500)
    model.fit(x_train, y_train)
    output_dir = Path(MODELS_DIR_PATH) / name_date_template
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / MODEL_FILENAME
    logger.info('saving model')
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
        logger.info('model saved')
    context['task_instance'].xcom_push(key='latest_model_path', value=str(output_path))


def _wait_3(**context):
    logger.info('waiting for train')
    latest_model_path = context['task_instance'].xcom_pull(task_ids='train', key='latest_model_path')
    return Path(latest_model_path).exists()


def _val(name_date_template, **context):
    val_path = Path(PROCESSED_DATA_PATH) / VAL_DATA_FILENAME
    val_df = pd.read_csv(val_path)
    logger.info('extract targets')
    x_val, y_val = val_df.drop('target', axis=1), val_df['target']
    logger.info('load model')
    latest_model_path = context['task_instance'].xcom_pull(task_ids='train', key='latest_model_path')
    with open(latest_model_path, 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(x_val)
    roc_path = Path(MODELS_DIR_PATH) / name_date_template / 'roc_auc.txt'
    roc_auc = roc_auc_score(y_val, y_pred)
    logger.info(f'achieved ROC AUC: {roc_auc}')
    with open(roc_path, 'w') as f:
        fpr, tpr, thr = roc_curve(y_val, y_pred)
        f.write(str(roc_auc))
        f.write(f'fpr:\n{fpr}\ntpr\n{tpr}\nthresholds\n{thr}')
    context['task_instance'].xcom_push(key='latest_roc_path', value=str(roc_path))


with DAG(dag_id='train_dag',
         schedule_interval='@weekly',
         start_date=days_ago(5)) as dag:
    wait_0 = PythonSensor(task_id='wait_0', python_callable=_wait_0)
    preprocess = PythonOperator(task_id='preprocess', python_callable=_preprocess)
    wait_1 = PythonSensor(task_id='wait_1', python_callable=_wait_1)
    split = PythonOperator(task_id='split', python_callable=_split)
    wait_2 = PythonSensor(task_id='wait_2', python_callable=_wait_2)
    train = PythonOperator(task_id='train',
                           python_callable=_train,
                           op_kwargs={'name_date_template': '{{ ds }}'})
    wait_3 = PythonSensor(task_id='wait_3', python_callable=_wait_3)
    val = PythonOperator(task_id='val',
                         python_callable=_val,
                         op_kwargs={'name_date_template': '{{ ds }}'})

    wait_0 >> preprocess >> wait_1 >> split >> wait_2 >> train >> wait_3 >> val

# if __name__ == '__main__':
#     _train('abc')
