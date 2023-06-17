import pickle
import pandas as pd
import sys

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(year: int, month: int):
    df = pd.read_parquet(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def apply_and_process_data(df):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    
    print("Predictions:")
    for pred in y_pred:
        print(pred)
    
    print("Mean Prediction:", y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['prediction'] = y_pred

    df_result.to_parquet(
        'output.parquet',
        engine='pyarrow',
        compression=None,
        index=False
    )
    
    return df

def run():
    month = int(sys.argv[1])  # 03
    year = int(sys.argv[2])  # 2022
    
    df = read_data(year=year, month=month)
    apply_and_process_data(df)

if __name__ == "__main__":
    run()
