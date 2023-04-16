# pylint: disable=E0401, R0913, R0914, W0703

"""
Module for extraction, sorting and plotting the features by their importance
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_and_plot_feature_importance(model, df_columns: [],
                                    num=200, plot_bar=False) -> pd.DataFrame:
    """
    Extracts, sorts and plots features by their importance for the model
    @param model: The model (LightGBM i.e.)
    @param df_columns: The list of the train dataframe columns
    @param num: The maximum number of features to be extracted
    @param plot_bar: Flag if the plot of featured by the importance is needed
    @return: The olist of features of the model sorted by their importance
    """
    _feature_imp = __get_feature_importance_data_frame(model, df_columns)
    if plot_bar:
        plt.figure(figsize=(21, num // 3))
        sns.set(font_scale=1)
        sns.barplot(x="Value", y="Feature", data=_feature_imp[0:num])
        plt.tight_layout()
        plt.show()
    return _feature_imp


def __filter_top_features(features_importance: list,
                          top_feature_count: int,
                          categorical_features: list) -> list:
    if top_feature_count <= 0:
        return features_importance
    feat_selected = features_importance[0:top_feature_count]
    return merge_categorical_features(feat_selected, categorical_features)


def merge_categorical_features(feat_selected: list, categorical_features: list) -> list:
    """
    Merges the list of the selected features with the categorical features
    @param feat_selected: The list of the selected features (columns)
    @param categorical_features: The list of the categorical features (columns)
    @return: The merged list
    """
    if categorical_features is None or len(categorical_features) == 0:
        return feat_selected
    for col in categorical_features:
        if col not in feat_selected:
            feat_selected.append(col)
    return feat_selected


def __get_feature_importance_data_frame(model, df_columns: []) -> pd.DataFrame:
    feature_imp = None
    if getattr(model, "feature_importances_", None) is not None:
        feature_imp = model.feature_importances_
    elif getattr(model, "feature_importance", None) is not None and callable(
            getattr(model, "feature_importance", None)):
        feature_imp = model.feature_importance()
    feature_imp_df = pd.DataFrame({'Value': feature_imp, 'Feature': df_columns})
    feature_imp_df.sort_values(by="Value", ascending=False, inplace=True)
    return feature_imp_df


def get_top_features_from_list(features_importance: list,
                               df_columns: [],
                               top_feature_count: int,
                               categorical_features: list = None) -> []:
    """

    @param features_importance: The feature importance list
    (i.e. extracted from a fitted model)
    @param df_columns: The list of the train dataframe columns
    @param top_feature_count: The number of the most important features to be returned
    @param categorical_features: The list of the categorical features of the model
    @return: The list of the first top_feature_count most important columns
    """
    feat_selected = __filter_top_features(features_importance, top_feature_count,
                                          categorical_features)
    top_features = df_columns.intersection(feat_selected)
    return top_features


def get_features_importance_list(model, df_columns: []) -> list:
    """
    Extract the list of features with the importance scores from the model
    @param model: The model (LightGBM or other)
    @param df_columns: The list of the train dataframe columns
    @return: The list of features with importance scores
    """
    _feature_importance = __get_feature_importance_data_frame(model, df_columns)
    _feature_imp_sorted = _feature_importance['Feature'].values.tolist()
    return _feature_imp_sorted

# def get_top_features_from_model(model, x_train, y_train, top_feature_count: int,
#                                 categorical_features: list = None,
#                                 model_is_fitted=False):
#     if not model_is_fitted:
#         model.fit(x_train, y_train)
#     features_importance = get_features_importance_list(model, x_train.columns)
#     feat_selected = __filter_top_features(features_importance,
#                                           top_feature_count, categorical_features)
#     top_features = x_train.columns.intersection(feat_selected)
#     return top_features
