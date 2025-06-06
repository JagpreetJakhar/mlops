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

def main():
    df = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet')
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    print("Mean:",df.duration.mean())
    print("Std Dev:",df.duration.std())

    print(len(df[(df.duration >= 1) & (df.duration <= 60)]) / len(df) * 100)
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    train_dicts = df[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    print(f'Feature matrix size: {X_train.shape}')

    target = 'duration'
    y_train = df[target].values
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)

    print(f'Train RMSE: {root_mean_squared_error(y_train, y_pred)}')

    categorical = ['PULocationID', 'DOLocationID']

    def read_data(filename):
        df = pd.read_parquet(filename)

        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].astype('str')
        
        return df
    df_val = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet')
    val_dicts = df_val[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_val = df_val.duration.values
    y_pred = lr.predict(X_val)
    print(f'Val RMSE: {root_mean_squared_error(y_val, y_pred)}')

if __name__=="__main__":
    main()

