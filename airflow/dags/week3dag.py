# mlflow_pipeline_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore')

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
mlflow.set_experiment("week3-mlops")

default_args = {
    'start_date': datetime(2024, 1, 1),
}

def train_model():
    categorical = ['PULocationID', 'DOLocationID']

    def read_dataframe(filename):
        df = pd.read_parquet(filename)
        print(f'Number of records before preprocessing:{len(df)}')
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        df[categorical] = df[categorical].astype(str)
        return df

    df = read_dataframe('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')
    print(f"Number of records after filtering: {len(df)}")

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    target = 'duration'
    y_train = df[target].values

    mlflow.autolog(log_datasets=False)
    with mlflow.start_run():
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_train)
        rmse = root_mean_squared_error(y_train, y_pred)
        print(f'Train RMSE: {rmse}')
        print(f'Model intercept: {lr.intercept_}')

with DAG('mlflow_training_pipeline',
         schedule=None,
         catchup=False,
         default_args=default_args,
         tags=['mlflow', 'training']
         ) as dag:

    train_task = PythonOperator(
        task_id='train_linear_model',
        python_callable=train_model
    )

    train_task

