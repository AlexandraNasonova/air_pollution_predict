"""
DVC Stage tune_tft_model - find best TemporalFusionTransformer model params using optuna
"""
# pylint: disable=E0401, R0913, W1514


from argparse import ArgumentParser
import warnings
import pandas as pd
import yaml
from settings import settings
import data_preprocessing.columns_filter as col_filter
from model_tune_helpers.dl.tft_data_converter import TemporaryFusionTransformerAdapter
from model_tune_helpers.models_saving.mlflow_adapter import MlFlowAdapter
from model_tune_helpers.models_saving import lightning_adapter, json_adapter

warnings.filterwarnings('ignore')

STAGE = "tune_tft_model"


# noinspection DuplicatedCode
def __parse_args():
    parser = ArgumentParser(STAGE)
    parser.add_argument('--input_file', required=True, help='Path to input data')
    parser.add_argument('--output_metrics_file', required=True, help='Path to metrics file')
    # parser.add_argument('--output_model_params_file', required=True, help='Path to model file')
    parser.add_argument('--output_checkpoint_file', required=True,
                        help='Path to best model checkpoint file')
    parser.add_argument('--params', required=True, help='Path to params')
    parser.add_argument('--params_section', required=True, help='Section with filter params')
    parser.add_argument('--mlflow_env_file', required=True, help='Path to env file MlFlow')
    # parser.add_argument('--mlflow_artifact', required=True, help='MLFlow model artifacts path')
    return parser.parse_args()


if __name__ == '__main__':
    print(f'Stage {STAGE} started')
    # read params
    stage_args = __parse_args()
    with open(stage_args.params, 'r', encoding='UTF-8') as file_stream:
        yaml_params = yaml.safe_load(file_stream)
    model_params = yaml_params[stage_args.params_section]
    metric = yaml_params["metric"]
    run_name = f'{model_params["exp_name"]}_{model_params["run_name"]}'
    target_column_name = col_filter.get_target_column(
        prediction_value_type=model_params['prediction_value_type'],
        pol_id=model_params["pol_id"])

    # init TemporaryFusionTransformer dataset
    tft_adapter = TemporaryFusionTransformerAdapter(
        dataset_params=model_params["dataset_params"],
        target_column=target_column_name)

    df = pd.read_csv(stage_args.input_file, parse_dates=True,
                     index_col=settings.DATE_COLUMN_NAME)
    tft_adapter.prepare_dataset(df=df)

    run_name = f'{model_params["exp_name"]}_{model_params["run_name"]}'
    # init MlFlow experiment and autolog
    mlflow_adapter = MlFlowAdapter(
        model_tp=MlFlowAdapter.ModelType.PD_ARIMA)
    mlflow_adapter.set_experiment(experiment_name=model_params["exp_name"],
                                  mlflow_env_file=stage_args.mlflow_env_file)
    mlflow_adapter.start_run(run_name=run_name)

    try:
        mlflow_adapter.save_extra_params(model_params, "pipeline/pipeline_params.yaml")
        print('----TFT train started---')
        tft_adapter.train(trainer_params=model_params["trainer_params"],
                          tft_params=model_params["tft_params"])
        print('----TFT train finished---')

        # mlflow_adapter.save_model(model=tft_adapter,
        #                           x_train_df=tft_adapter.get_train_dataloader(),
        #                           y_train_df=tft_adapter.get_val_dataloader(),
        #                           artifact_path=stage_args.params_section,
        #                           best_model_params={
        #                               "trainer_params": model_params["trainer_params"],
        #                               "tft_params": model_params["tft_params"]
        #                           }
        #                           )
        mlflow_adapter.save_params({
              "trainer_params": model_params["trainer_params"],
              "tft_params": model_params["tft_params"]
          })
        lightning_adapter.copy_best_checkpoint(
            tft_adapter.get_best_model_path(), stage_args.output_checkpoint_file)
        mlflow_adapter.save_artifact(tft_adapter.get_best_model_path(),
                                     artifact_path="pkl")
        print(f'---Model is saved')

        val_score_best = tft_adapter.get_val_metric()
        print(f'---Model trained with best params: '
              f'best_val_score: {val_score_best}')

        mlflow_adapter.save_metrics(train_score=0,
                                    val_score=val_score_best,
                                    metric_name=metric)
        json_adapter.save_metrics_to_json(
            file_path=stage_args.output_metrics_file,
            train_score=0, val_score=val_score_best, metric_name=metric)
        print(f'---Metrics are saved')

        print(f'Stage {STAGE} finished')
    finally:
        pass
        mlflow_adapter.end_run()
