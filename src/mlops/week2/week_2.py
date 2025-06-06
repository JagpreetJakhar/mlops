import mlflow
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import prettytable
import warnings
warnings.filterwarnings('ignore')

mlflow.set_experiment("week2-mlops")
import os

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
def main(): 
    categorical = ['PULocationID', 'DOLocationID']
    def read_dataframe(filename):
        df = pd.read_parquet(filename)

        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)]
        df[categorical] = df[categorical].astype(str)
        
        return df
    df = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')
    print(f"Number of records before filtering: {len(df)}")
    df = read_dataframe('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')
    print(df.columns)
    print(f"Number of records after filtering: {len(df)}")
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    print(f'Feature matrix size: {X_train.shape}')
    target = 'duration'
    y_train = df[target].values
    mlflow.autolog(log_datasets=False)
    with mlflow.start_run():
        lr = LinearRegression()
        #mlflow.set_tag("model","Linear Regresssion")
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_train)
        error = root_mean_squared_error(y_train,y_pred)
        #mlflow.log_metric("rmse",error)
        print(f'Train RMSE: {root_mean_squared_error(y_train, y_pred)}')
        print(f'Model intercept: {lr.intercept_}')
if __name__=="__main__":
    main()
