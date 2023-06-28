import datetime
import mlflow
from lightgbm import LGBMClassifier
from sklearn import datasets
from settings import env_reader


def print_auto_logged_info(run):
    tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [
        f.path for f in mlflow.MlflowClient().list_artifacts(run.info.run_id, "model")
    ]
    feature_importances = [
        f.path
        for f in mlflow.MlflowClient().list_artifacts(run.info.run_id)
        if f.path != "model"
    ]
    print(f"run_id: {run.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"feature_importances: {feature_importances}")
    print(f"params: {run.data.params}")
    print(f"metrics: {run.data.metrics}")
    print(f"tags: {tags}")


if __name__ == "__main__":
    env_values = env_reader.get_params()
    mlflow.set_tracking_uri(f'{env_values.get("MLFLOW_URI")}:{env_values.get("MLFLOW_PORT")}')
    mlflow.set_experiment(f'lgbm_{datetime.datetime.now().strftime("%d_%m_%Y_T_%H_%M_%S")}')

    X, y = datasets.load_iris(return_X_y=True, as_frame=True)
    model = LGBMClassifier(objective="multiclass", random_state=42)
    # Auto log all MLflow entities
    mlflow.lightgbm.autolog()

    # Train the model
    with mlflow.start_run() as run:
        model.fit(X, y)

    # fetch the auto logged parameters and metrics
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
