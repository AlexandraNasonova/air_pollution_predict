import datetime
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from settings import env_reader

if __name__ == "__main__":
    env_values = env_reader.get_params()
    mlflow.set_tracking_uri(f'{env_values.get("MLFLOW_URI")}:{env_values.get("MLFLOW_PORT")}')
    mlflow.set_experiment(datetime.datetime.now().strftime('%d_%m_%Y_T_%H_%M_%S'))

    with mlflow.start_run() as run:
        mlflow.autolog()

        db = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        mse = mean_squared_error(y_true=y_test, y_pred=predictions)
        print(f'MSE: {mse}')
        mlflow.log_metric("MSE", mse)

        #artifacts
        artifact_uri = mlflow.get_artifact_uri()
        print(artifact_uri)
        mlflow.log_dict({"mlflow-version": "0.28", "n_cores": "10"}, "config.json")
        config_json = mlflow.artifacts.load_dict(artifact_uri + "/config.json")
        print(config_json)

        print('---End---')