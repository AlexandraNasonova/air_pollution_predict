import datetime
from sklearn.ensemble import RandomForestRegressor

from settings import env_reader
from sklearn.datasets import load_diabetes

import mlflow
from mlflow import MlflowClient


def yield_artifacts(run_id, path=None):
    """Yield all artifacts in the specified run"""
    client = MlflowClient()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from yield_artifacts(run_id, item.path)
        else:
            yield item.path


if __name__ == "__main__":
    env_values = env_reader.get_params()
    mlflow.set_tracking_uri(f'{env_values.get("MLFLOW_URI")}:{env_values.get("MLFLOW_PORT")}')
    mlflow.set_experiment(f'lgbm_{datetime.datetime.now().strftime("%d_%m_%Y_T_%H_%M_%S")}')

    num_rows = 50
    X, y = dataset = load_diabetes(return_X_y=True, as_frame=True)
    X = X[:50]
    y = y[:50]
    X_cast = X.astype('float32')  # training set cast to 32bit float
    X_cast, y = X_cast.iloc[:num_rows], y[:num_rows]
    X = X.iloc[:num_rows]
    X = X.astype('float64')  # SHAP kernel explainer on 64bit float

    model = RandomForestRegressor()
    model.fit(X_cast, y)  # fit on 32bit float

    with mlflow.start_run() as run:
        explanation_uri = mlflow.shap.log_explanation(model.predict, X)

    artifact_path = "model_explanations_shap"
    artifacts = set(yield_artifacts(run.info.run_id))

    # assert explanation_uri == os.path.join(run.info.artifact_uri, artifact_path)
    # assert artifacts == {
    #     os.path.join(artifact_path, "base_values.npy"),
    #     os.path.join(artifact_path, "shap_values.npy"),
    #     os.path.join(artifact_path, "summary_bar_plot.png"),
    # }

    # explainer = shap.KernelExplainer(model.predict, shap.kmeans(X, num_rows))
    # shap_values_expected = explainer.shap_values(X)
    #
    # dst_path = client.download_artifacts(run.info.run_id, artifact_path)
    # base_values = np.load(os.path.join(dst_path, "base_values.npy"))
    # shap_values = np.load(os.path.join(dst_path, "shap_values.npy"))

    # base_values = np.load(os.path.join(explanation_uri, "base_values.npy"))
    # shap_values = np.load(os.path.join(explanation_uri, "shap_values.npy"))
    # np.testing.assert_array_equal(base_values, explainer.expected_value)
    # np.testing.assert_array_equal(shap_values, shap_values_expected)



    # # prepare training data
    # X, y = dataset = load_diabetes(return_X_y=True, as_frame=True)
    # X = X[:50]
    # y = y[:50]
    # # X = pd.DataFrame(dataset.data[:50, :8], columns=dataset.feature_names[:8])
    # # y = dataset.target[:50]
    #
    # # train a model
    # model = LinearRegression()
    # model.fit(X, y)
    #
    # # log an explanation
    # with mlflow.start_run() as run:
    #     mlflow.shap.log_explanation(model.predict, X)
    #
    # # list artifacts
    # client = MlflowClient()
    # artifact_path = "model_explanations_shap"
    # artifacts = [x.path for x in client.list_artifacts(run.info.run_id, artifact_path)]
    # print("# artifacts:")
    # print(artifacts)
    #
    # # load back the logged explanation
    # dst_path = client.download_artifacts(run.info.run_id, artifact_path)
    # base_values = np.load(os.path.join(dst_path, "base_values.npy"))
    # shap_values = np.load(os.path.join(dst_path, "shap_values.npy"))
    #
    # print("\n# base_values:")
    # print(base_values)
    # print("\n# shap_values:")
    # print(shap_values[:3])