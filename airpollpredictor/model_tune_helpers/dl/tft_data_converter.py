import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import settings.settings as settings
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import RMSE, QuantileLoss


class TemporaryFusionTransformerAdapter:
    def __init__(self, dataset_params: {}, target_column):
        self._train_dataloader = None
        self._val_dataloader = None
        self._training = None
        self._trainer = None

        self._early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-4,
            patience=10, verbose=False, mode="min")
        self._lr_logger = LearningRateMonitor()
        self._logger = TensorBoardLogger("lightning_logs")

        self._dataset_params = dataset_params
        self._target_column = target_column

    def get_train_dataloader(self):
        return self._train_dataloader

    def get_val_dataloader(self):
        return self._val_dataloader

    def get_val_trainer(self):
        return self._trainer

    def get_best_model_path(self) -> str:
        return self._trainer.checkpoint_callback.best_model_path

    @staticmethod
    def __index_adjust(df: pd.DataFrame):
        df.reset_index(inplace=True)
        df[settings.DATE_COLUMN_NUM_IND_NAME] = \
            (df[settings.DATE_COLUMN_NAME]
             - df[settings.DATE_COLUMN_NAME].min()).dt.days
        df.drop(columns=[settings.DATE_COLUMN_NAME], inplace=True)

    def __target_to_float(self, df: pd.DataFrame):
        df[self._target_column] = df[self._target_column].astype(float)

    def __columns_to_categories(self, df: pd.DataFrame):
        for col in self._dataset_params["time_varying_known_categoricals"]:
            df[col] = df[col].astype(str).astype("category")
        for col in self._dataset_params["group_ids"]:
            df[col] = df[col].astype(str).astype("category")

    @staticmethod
    def __merge_train_val(df_train: pd.DataFrame, df_val: pd.DataFrame):
        df_val[settings.DATE_COLUMN_NUM_IND_NAME] += \
            df_train[settings.DATE_COLUMN_NUM_IND_NAME].max() + 1 \
            - df_val[settings.DATE_COLUMN_NUM_IND_NAME].min()
        df = pd.concat([df_train, df_val], ignore_index=True)
        return df

    def __set_timeseries_dataset(self, df: pd.DataFrame):
        max_prediction_length = self._dataset_params["max_prediction_length"]
        max_encoder_length = self._dataset_params["max_encoder_length"]
        training_cutoff = df[settings.DATE_COLUMN_NUM_IND_NAME].max() - max_prediction_length

        time_varying_known_reals = [settings.DATE_COLUMN_NUM_IND_NAME]
        if self._dataset_params["time_varying_known_reals"]:
            time_varying_known_reals += self._dataset_params["time_varying_known_reals"]

        self._training = TimeSeriesDataSet(
            df[lambda x: x[settings.DATE_COLUMN_NUM_IND_NAME] <= training_cutoff],
            time_idx=settings.DATE_COLUMN_NUM_IND_NAME,
            target=self._target_column,
            group_ids=self._dataset_params["group_ids"],
            min_encoder_length=max_encoder_length // 2,
            # keep encoder length long (as it is in the validation set)
            max_encoder_length=max_encoder_length,  # lookback period
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            # static_categoricals=params["group_ids"],
            static_reals=[] ,#self._dataset_params["static_reals"],
            time_varying_known_categoricals=self._dataset_params["time_varying_known_categoricals"],
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[],
            target_normalizer=GroupNormalizer(
                groups=self._dataset_params["group_ids"], transformation="softplus"
            ),  # use softplus and normalize by group
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        # create validation set (predict=True) which means to predict the last max_prediction_length points in time
        # for each series
        validation = TimeSeriesDataSet.from_dataset(self._training, df, predict=True, stop_randomization=True)
        # create dataloaders for model
        batch_size = self._dataset_params["batch_size"]  # set this 32 to 128
        self._train_dataloader = self._training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        self._val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    # def prepare_dataset(self, df_train: pd.DataFrame, df_val: pd.DataFrame):
    #     df_train_c = df_train.copy(deep=True)
    #     df_val_c = df_val.copy(deep=True)
    #     self.__index_adjust(df_train_c)
    #     self.__index_adjust(df_val_c)
    #     self.__columns_to_categories(df_train_c)
    #     self.__columns_to_categories(df_val_c)
    #     df = self.__merge_train_val(df_train_c, df_val_c)
    #     self.__set_timeseries_dataset(df)

    def prepare_dataset(self, df: pd.DataFrame):
        df_c = df.copy(deep=True)
        self.__target_to_float(df_c)
        self.__columns_to_categories(df_c)
        self.__index_adjust(df_c)
        self.__set_timeseries_dataset(df_c)

    def train(self, trainer_params, tft_params):
        pl.seed_everything(42)

        self._trainer = pl.Trainer(
            accelerator="cpu",
            enable_model_summary=False,
            callbacks=[self._lr_logger, self._early_stop_callback],
            logger=self._logger,
            **trainer_params
        )
        tft = TemporalFusionTransformer.from_dataset(
            self._training,
            loss=QuantileLoss(),
            **tft_params
        )
        print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")
        self._trainer.fit(
            tft,
            train_dataloaders=self._train_dataloader,
            val_dataloaders=self._val_dataloader,
        )

    def get_predictions(self):
        best_model_path = self._trainer.checkpoint_callback.best_model_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        predictions = best_tft.predict(self._val_dataloader,
                                       return_y=True,
                                       trainer_kwargs=dict(accelerator="cpu"))
        return predictions

    def get_val_metric(self):
        predictions = self.get_predictions()
        rmse = RMSE()(predictions.output, predictions.y).item()
        return rmse, predictions
