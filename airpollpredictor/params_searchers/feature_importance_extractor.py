import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance(model, df_columns, num=200, plot_bar=False):
    _feature_imp = get_feature_importance_data_frame(model, df_columns)
    if plot_bar:
        plt.figure(figsize=(21, num // 3))
        sns.set(font_scale=1)
        sns.barplot(x="Value", y="Feature", data=_feature_imp[0:num])
        plt.tight_layout()
        plt.show()
    return _feature_imp


def filter_top_features(features_importance, top_feature_count, categorical_features) -> list:
    if top_feature_count <= 0:
        return features_importance
    feat_selected = features_importance[0:top_feature_count]
    return merge_categorical_features(feat_selected, categorical_features)


def merge_categorical_features(feat_selected, categorical_features) -> list:
    if categorical_features is None or len(categorical_features) == 0:
        return feat_selected
    for col in categorical_features:
        if col not in feat_selected:
            feat_selected.append(col)
    return feat_selected


def get_feature_importance_data_frame(model, df_columns) -> pd.DataFrame:
    feature_imp = None
    if getattr(model, "feature_importances_", None) is not None:
        feature_imp = model.feature_importances_
    elif getattr(model, "feature_importance", None) is not None and callable(
            getattr(model, "feature_importance", None)):
        feature_imp = model.feature_importance()
    feature_imp_df = pd.DataFrame({'Value': feature_imp, 'Feature': df_columns})
    feature_imp_df.sort_values(by="Value", ascending=False, inplace=True)
    return feature_imp_df


def get_features_importance_list(model, df_columns) -> list:
    _feature_importance = get_feature_importance_data_frame(model, df_columns)
    _feature_imp_sorted = _feature_importance['Feature'].values.tolist()
    return _feature_imp_sorted


def get_top_features_from_list(features_importance: list, df_columns, top_feature_count: int,
                               categorical_features: list = None):
    feat_selected = filter_top_features(features_importance, top_feature_count, categorical_features)
    top_features = df_columns.intersection(feat_selected)
    return top_features


def get_top_features_from_model(model, x_train, y_train, top_feature_count: int, categorical_features: list = None,
                                model_is_fitted=False):
    if not model_is_fitted:
        model.fit(x_train, y_train)
    features_importance = get_features_importance_list(model, x_train.columns)
    feat_selected = filter_top_features(features_importance, top_feature_count, categorical_features)
    top_features = x_train.columns.intersection(feat_selected)
    return top_features
