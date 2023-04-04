import joblib
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


### Functions to load timeseries
def ts_loader(timeseries) -> {Copy}:
    ts_final = pd.read_csv(timeseries, low_memory=False)
    ts_final['DatetimeEnd'] = pd.to_datetime(ts_final['DatetimeEnd'], format="%Y-%m-%d %H:%M:%S")
    ts_final = ts_final.set_index('DatetimeEnd')
    return ts_final

### Functions to generate lag-features
### Generates only numeric features
def feature_creator_v1(df_temp,number_of_lag_features=30) -> {Copy}:
    df = df_temp.copy()
    df.columns = ['aqi_d0']
    n_features = int(number_of_lag_features)
    current_column_name = 'aqi_d0'
    prev_column_name = 'aqi_d0'
    iternum = 1
    for i in range(n_features):
        current_column_name = current_column_name[:5] + str(iternum)
        df[current_column_name] = df[prev_column_name].shift(1)
        iternum +=1
        prev_column_name = current_column_name
    ### Truncate first "n" rows to remove rows containing NaN values
    df = df.iloc[number_of_lag_features:, :]
    return df

### Functions to generate categorical and lag-features
### Generates both categorical and numeric features
def feature_creator_v2(df_temp,number_of_lag_features=30) -> {Copy}:
    df = df_temp.copy()
    df.columns = ['aqi_d0']
    df['month'] = df.index.month
    df['monthday'] = df.index.strftime("%d")
    df['weekday'] = df.index.weekday
    weekdays_list = list(df["weekday"])
    weekend_list = []
    for i in range(len(weekdays_list)):
        weekend_flag = False
        if (weekdays_list[i] == 6) | (weekdays_list[i] == 0):
            weekend_flag = True
        weekend_list.append(weekend_flag)
    df['weekend'] = weekend_list
    n_features = int(number_of_lag_features)
    current_column_name = 'aqi_d0'
    prev_column_name = 'aqi_d0'
    iternum = 1
    for i in range(n_features):
        current_column_name = current_column_name[:5] + str(iternum)
        df[current_column_name] = df[prev_column_name].shift(1)
        iternum +=1
        prev_column_name = current_column_name
    ### Truncate first "n" rows to remove rows containing NaN values
    df = df.iloc[number_of_lag_features:, :]
    return df

### Function to split dataset into train and test parts
### Splitby must be string in the following format: YEAR-MONTH-DAY
### month and day must have leading zero for values less than 10
def train_test_splitter(df_temp, splitby='2022-01-01') -> tuple[Any, Any]:
    df = df_temp
    cutoff = splitby + ' 00:00:00+01:00'
    df_train = df[df.index < cutoff].copy()
    df_test = df[df.index >= cutoff].copy()
    return df_train, df_test

### Function to split data into X(features) and y(target) components
def xy_splitter(df_train) -> tuple[Any, Any]:
    target_column = 'aqi_d0'
    X_train = df_train.drop([target_column], axis=1)
    y_train = df_train[target_column]
    return X_train, y_train

### Function to split data into X(features) and y(target) components
def xy_splitter_v2(df_train, df_test) -> tuple[Any, Any, Any, Any]:
    target_column = 'aqi_d0'
    X_train = df_train.drop([target_column], axis=1)
    X_test = df_test.drop([target_column], axis=1)
    y_train = df_train[target_column]
    y_test = df_test[target_column]
    return X_train, X_test, y_train, y_test

### Function to initialize and train linear regression model
def linreg_model(X_train, y_train) -> None:
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, 'scaler.pkl')
    X_train_scaled = scaler.transform(X_train)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    joblib.dump(lr, 'model.pkl')
    return

### Function to make predictions using pickled model
### Accepts pandas dataframe
### Returns a copy of data with "predictions" column
def model_predict(X_test_data, model_path, scaler_path='None') -> {Copy}:
    X_test = X_test_data.copy()
    if scaler_path !='None':
        scaler = joblib.load(scaler_path)
        X_test_ready = scaler.transform(X_test_data)
    else:
        X_test_ready = X_test_data
    model = joblib.load(model_path)
    y_pred = model.predict(X_test_ready)
    X_test['prediction'] = y_pred
    return X_test

def pipeline(trainfile, testfile, outputfile):
    # noinspection PyArgumentList
    linreg_model(xy_splitter(feature_creator_v1(ts_loader(trainfile))))
    predictions = model_predict(testfile, 'model.pkl', scaler_path='scaler.pkl')
    predictions.to_csv(outputfile, index=False, encoding='utf-8-sig')

def pipeline_v2(trainfile, train_splitby, outputfile):
    # noinspection PyArgumentList
    X_train, X_test, y_train, _ = xy_splitter_v2(train_test_splitter(feature_creator_v1(ts_loader(trainfile)), splitby=train_splitby))
    linreg_model(X_train, y_train)
    predictions = model_predict(X_test, 'model.pkl', scaler_path='scaler.pkl')
    predictions.to_csv(outputfile, index=False, encoding='utf-8-sig')
    return

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainfile", type=str, required=True, help="path to train data")
    parser.add_argument("--train_splitby", type=str, required=False, help="date to split data")
    parser.add_argument("--testfile", type=str, required=False, help="path to test data")
    parser.add_argument("--outputfile", type=str, required=True, help="path to output data")
    args = parser.parse_args()
    if not args.train_splitby:
        pipeline(args.trainfile, args.testfile, args.outputfile)
        print(f'done')
        return
    pipeline_v2(args.trainfile, args.train_splitby, args.outputfile)
    print(f'done')
    return

if __name__ == "__main__":
    main()