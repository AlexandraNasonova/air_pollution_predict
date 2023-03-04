import lightgbm as lgb
import optuna as optuna
# import optuna.integration.lightgbm as optuna_lgb
from lightgbm import early_stopping
from lightgbm import log_evaluation
import ml_models_search.params_searchers.feature_importance_extractor as feat_imp


class OptunaLgbSearch:
    study: optuna.Study
    study_best_params: {}

    features_importance: []

    best_train_score: float
    best_val_score: float
    best_model: lgb.Booster
    best_features_count: int
    best_features_list: []

    # tuner_best_score: float
    # tuner_best_params: dict
    # tuner_categorical_feature: []

    def __init__(self, study_name: str, objective: str, metric: str, x_train, y_train, x_val, y_val,
                 default_params=None, default_category=None, categories_for_optimization=None,
                 default_top_features_count=-1):
        self.study_name = study_name
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.default_category = default_category
        if categories_for_optimization is None:
            categories_for_optimization = []
        self.categories = categories_for_optimization
        if default_category is not None and not (default_category in categories_for_optimization):
            self.categories.append(default_category)
        self.objective_lgbm = objective
        self.metric = metric
        if default_params != {}:
            self.default_params = default_params
        else:
            self.default_params = {}

        self.study_best_params = None
        self.best_categorical_feature = None
        if default_top_features_count > 0:
            self.set_best_features_by_count(default_top_features_count)
        else:
            self.set_best_features_by_count(x_train.shape[0])


    def get_model_params_from_trial(self, trial):
        model_params = {
            'n_jobs': trial.suggest_categorical("n_jobs", [-1]),
            'verbosity': trial.suggest_categorical("verbosity", [-1]),
            'objective': trial.suggest_categorical("objective", [self.objective_lgbm]),
            'metric': trial.suggest_categorical("metric", [self.metric]),
            'boosting_type': trial.suggest_categorical("boosting_type", ['gbdt']),
            # 'boosting_type': trial.suggest_categorical("boosting_type", ['gbdt', 'goss', 'dart']),

            'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
            'n_estimators': trial.suggest_int('n_estimators', 1000, 3000, 100),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.16),
            'subsample': trial.suggest_float('subsample', 0, 1),
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 10),
            'subsample_for_bin': trial.suggest_int('subsample_for_bin', 50_000, 500_000),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 600),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'max_bin': trial.suggest_int('max_bin', 50, 400),
            # 'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1),
        }
        model_params['subsample'] = 1.0 if model_params['boosting_type'] == 'goss' else model_params['subsample']
        return model_params

    def set_feature_importance(self, params=None):
        if params is not None and params != {}:
            model_params = params
        elif self.study_best_params is not None and self.study_best_params != {}:
            model_params = self.study_best_params
        else:
            model_params = self.default_params
        categorical_features = self.default_category

        train_score, val_score, model = \
            self.run_model_and_eval(params=model_params, categorical_features=categorical_features)
        self.features_importance = feat_imp.get_features_importance_list(model, self.x_train.columns)

    def objective_feature(self, trial, params=None):
        if params is not None and params != {}:
            model_params = params
        elif self.study_best_params is not None and self.study_best_params != {}:
            model_params = self.study_best_params
        else:
            model_params = self.default_params

        top_feature_count = trial.suggest_int('top_feature_count', 100, len(self.features_importance))
        categorical_features = self.default_category
        top_features = feat_imp \
            .get_top_features_from_list(self.features_importance, self.x_train.columns,
                                        top_feature_count, categorical_features)
        x_train_f = self.x_train[top_features]
        x_val_f = self.x_val[top_features]

        metric_train, metric_val, model = \
            self.run_model_and_eval_per_df(model_params, categorical_features=categorical_features,
                                           _x_train=x_train_f, _y_train=self.y_train, _x_val=x_val_f,
                                           _y_val=self.y_val)

        return metric_val

    def objective(self, trial, search_category=True, with_pruner=False, best_features_only=False, cv_splitter=None):
        model_params = self.get_model_params_from_trial(trial)
        if search_category:
            categorical_features = trial.suggest_categorical('categorical_features', self.categories)
        else:
            categorical_features = self.default_category
        pruning_callback = None
        if with_pruner:
            pruning_callback = optuna.integration.LightGBMPruningCallback(trial, self.metric)

        if cv_splitter is None:
            metric_train, metric_val, model = \
                self.run_model_and_eval(model_params, categorical_features=categorical_features,
                                        best_features_only=best_features_only,
                                        pruning_callback=pruning_callback)
        else:
            metric_val = \
                self.run_model_cv(params=model_params, categorical_features=categorical_features,
                                  cv_splitter=cv_splitter, pruning_callback=pruning_callback,
                                  best_features_only=best_features_only)
        return metric_val

    @staticmethod
    def logging_callback(study, frozen_trial):
        previous_best_value = study.user_attrs.get("previous_best_value", None)
        if previous_best_value != study.best_value:
            study.set_user_attr("previous_best_value", study.best_value)
            print(
                "Trial {} finished with best value: {} and parameters: {}. ".format(
                    frozen_trial.number,
                    frozen_trial.value,
                    frozen_trial.params,
                )
            )

    def run_params_search(self, n_trials=10, n_jobs=1, study_name=None, direction='minimize',
                          cv_splitter=None, with_pruner=False, search_category=False, best_features_only=True,
                          save_best_params=True, warm_params=None):
        print(f'run_params_search n_trials={n_trials}, search_category={search_category}, '
              f'best_features_only={best_features_only}, with_pruner={with_pruner}')

        self.study = optuna.create_study(study_name=study_name if study_name is not None else self.study_name,
                                         pruner=None if not with_pruner else optuna.pruners.MedianPruner(
                                             n_warmup_steps=10),
                                         direction=direction)

        if warm_params is not None:
            self.study.enqueue_trial(warm_params)

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        self.study.optimize(
            (lambda trial:
             self.objective(trial, with_pruner=with_pruner, search_category=search_category,
                            best_features_only=best_features_only, cv_splitter=cv_splitter)),
            n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True,
            callbacks=[self.logging_callback], gc_after_trial=(n_jobs == -1 or n_jobs > 10))
        if save_best_params:
            self.study_best_params = self.study.best_params
        return self.study.best_value

    def run_features_search(self, n_trials=10, n_jobs=1, study_name=None, direction='minimize',
                            params=None, save_best_params=True):
        print(f'run_features_search n_trials={n_trials}')
        self.study = optuna.create_study(study_name=study_name if study_name is not None else self.study_name,
                                         direction=direction)
        self.set_feature_importance(params=params)
        self.study.optimize(
            (lambda trial: self.objective_feature(trial, params=params)),
            n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
        if save_best_params:
            self.best_features_count = self.study.best_params['top_feature_count']
            self.best_features_list = self.features_importance[0:self.best_features_count]
        return self.study.best_value

    def run_model_and_eval(self, params=None, categorical_features=None, pruning_callback=None, best_features_only=False,
                           best_features_list=None, set_as_best_model=False) -> (
            float, float, lgb.Booster):

        if params is None:
            if self.study_best_params is None:
                params = self.default_params
            else:
                params = self.study_best_params
        if categorical_features is None:
            if self.best_categorical_feature is None:
                categorical_features = self.default_category
            else:
                categorical_features = self.best_categorical_feature

        if best_features_only:
            if best_features_list is None:
                best_features_list = self.best_features_list
            top_features = feat_imp.merge_categorical_features(best_features_list, categorical_features)
            x_train_f = self.x_train[top_features]
            x_val_f = self.x_val[top_features]
        else:
            x_train_f = self.x_train
            x_val_f = self.x_val

        return self.run_model_and_eval_per_df(params=params, categorical_features=categorical_features,
                                              _x_train=x_train_f, _y_train=self.y_train, _x_val=x_val_f,
                                              _y_val=self.y_val, pruning_callback=pruning_callback,
                                              set_as_best_model=set_as_best_model)

    def run_model_and_eval_per_df(self, params, categorical_features, _x_train, _y_train, _x_val, _y_val,
                                  pruning_callback=None, set_as_best_model=False) -> (float, float, lgb.Booster):
        lgb_train = lgb.Dataset(data=_x_train, label=_y_train, categorical_feature=categorical_features)
        lgb_eval = lgb.Dataset(data=_x_val, label=_y_val, categorical_feature=categorical_features,
                               reference=lgb_train)
        evals_result = {}
        callbacks = [early_stopping(100, verbose=False), log_evaluation(0)]

        if pruning_callback is not None:
            callbacks.append(pruning_callback)

        model = lgb.train(params, lgb_train,
                          num_boost_round=50,
                          categorical_feature=categorical_features,
                          valid_sets=[lgb_train, lgb_eval],
                          valid_names=["train", "val"],
                          verbose_eval=False,
                          evals_result=evals_result,
                          callbacks=callbacks
                          )

        train_score = evals_result['train'][self.metric][-1]
        val_score = evals_result['val'][self.metric][-1]
        if set_as_best_model:
            self.save_best_model(model, categorical_features, train_score, val_score)
        return train_score, val_score, model

    def run_model_cv_per_df(self, params, categorical_features, _x_train, _y_train,
                            cv_splitter, pruning_callback=None) -> float:
        lgb_train = lgb.Dataset(data=_x_train, label=_y_train, categorical_feature=categorical_features)
        callbacks = [early_stopping(100, verbose=False), log_evaluation(0)]

        if pruning_callback is not None:
            callbacks.append(pruning_callback)

        cv_res_obj = lgb.cv(params, lgb_train, metrics=self.metric,
                            num_boost_round=100, folds=cv_splitter,
                            verbose_eval=False, callbacks=callbacks,
                            stratified=False, shuffle=False)

        val_score = cv_res_obj[f'{self.metric}-mean'][-1]
        return val_score

    def run_model_cv(self, params, cv_splitter, categorical_features=None, pruning_callback=None,
                     best_features_list=None, best_features_only=False) -> float:

        if categorical_features is None:
            categorical_features = self.default_category

        if best_features_only:
            if best_features_list is None:
                best_features_list = self.best_features_list
            top_features = feat_imp.merge_categorical_features(best_features_list, categorical_features)
            x_train_f = self.x_train[top_features]
        else:
            x_train_f = self.x_train

        return self.run_model_cv_per_df(params=params, categorical_features=categorical_features,
                                        _x_train=x_train_f, _y_train=self.y_train,
                                        cv_splitter=cv_splitter, pruning_callback=pruning_callback)

    def predict_by_best_model(self, x_test):
        top_features = feat_imp.merge_categorical_features(self.best_features_list, self.best_categorical_feature)
        x_test_f = x_test[top_features]
        return self.best_model.predict(x_test_f)

    def set_best_features_by_count(self, top_feature_count):
        self.set_feature_importance()
        if top_feature_count > 0:
            self.best_features_count = top_feature_count
        else:
            top_feature_count = len(self.x_train.columns)
        self.best_features_list = self.features_importance[0:top_feature_count]

    def save_best_model(self, model, categorical_features, train_score, val_score):
        self.best_model = model
        self.best_categorical_feature = categorical_features
        self.best_train_score = train_score
        self.best_val_score = val_score


    # def run_tuner_cv(self, cv_splitter, params=None, categorical_features=None, best_features_only=False,
    #                  num_boost_round=100, lgbm_early_stop_rounds=50, optuna_early_stop_rounds=30):
    #     if categorical_features is None:
    #         categorical_features = self.default_category
    #     if params is None:
    #         params = self.default_params
    #     lgb_train = lgb.Dataset(self.x_train, self.y_train, categorical_feature=categorical_features)
    #     tuner = optuna_lgb.LightGBMTunerCV(
    #         params,
    #         lgb_train,
    #         num_boost_round=num_boost_round,
    #         stratified=False, shuffle=False,
    #         early_stopping_rounds=lgbm_early_stop_rounds,
    #         show_progress_bar=True,
    #         folds=cv_splitter,
    #         callbacks=[early_stopping(optuna_early_stop_rounds), log_evaluation(100)]
    #     )
    #     tuner.run()
    #     self.tuner_best_score = tuner.best_score
    #     self.tuner_best_params = tuner.best_params
    #     self.tuner_categorical_feature = categorical_features
