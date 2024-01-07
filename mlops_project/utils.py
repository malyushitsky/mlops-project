import io
import logging
import os
import pickle

import dvc.api
import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from mlflow import log_metric, log_params
from mlflow.models import infer_signature
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)


def get_data(file, url):
    """
    func reads data from dvc and return it as pandas.DataFrame

    :param file: url to .dvc of file in project
    :param url: url to GitHub project
    :return: pandas.DataFrame
    """

    data = dvc.api.read(file, repo=url)
    df = pd.read_csv(io.StringIO(data), sep=",")
    return df


def preprocess_data(df, mode):
    """
    func gets dataset, train and apply imputers for columns
    and return preprocessed dataframe
    if mode == 'train' func return preprocessed dataframe
    with learned imputers and cat cols for Catboost
    elif mode == 'infer' it return only preprocessed dataframe

    :param df: dataframe needs to be preprocessed
    :param mode: function operating mode
    :return: if mode == 'train' -> pandas.DataFrame, SimpleImputer,
                                   SimpleImputer, List(Categorical_columns)
           elif mode == 'infer' -> pandas.DataFrame
    """

    cat_cols = df.select_dtypes(include=["object"]).columns.to_list()
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.to_list()

    if mode == "train":
        imputer_cat = SimpleImputer(strategy="most_frequent")
        imputer_num = SimpleImputer(strategy="median")
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
        df[num_cols] = imputer_num.fit_transform(df[num_cols])
        return df, imputer_cat, imputer_num, cat_cols

    elif mode == "infer":
        with open("../models/categorical_imputer.pickle", "rb") as handle:
            imputer_cat = pickle.load(handle)
        with open("../models/numerical_imputer.pickle", "rb") as handle:
            imputer_num = pickle.load(handle)

        df[cat_cols] = imputer_cat.transform(df[cat_cols])
        df[num_cols] = imputer_num.transform(df[num_cols])
        return df


def train_model(params, X_train, X_val, y_train, y_val, cat_cols):
    """
    func train catboost model with specific hyperparams

    :param params: hyperparams for catboost from hydra config
    :param X_train: X_train set
    :param X_val: X_val set
    :param y_train: y_target set
    :param y_val: y_val set
    :param cat_cols: List(categorical columns)
    :return: trained CatboostClassifier()
    """

    model = CatBoostClassifier(train_dir=None, allow_writing_files=False)
    model.set_params(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        verbose=False,
        plot=False,
        use_best_model=True,
        cat_features=cat_cols,
    )

    return model


def calc_metrics_with_logging(model, X, y, mode):
    """
    func calculate logloss and roc_auc for validation set and logs it
    into last run_id from mlfllow and hydra

    :param model: trained model
    :param X: X set
    :param y: y set
    :param mode: function operating mode
    :return: if mode == 'train' -> None,
           elif mode == 'infer' -> pd.Series (predictions)
    """

    mlflow.set_tracking_uri("http://128.0.1.1:8080")
    if mode == "train":
        with mlflow.start_run(run_name="TrainRun"):
            eval_metrics = model.evals_result_["learn"]
            log.info(f"Logloss_val: {eval_metrics['Logloss']}")
            log.info(f"Roc_auc_val: {eval_metrics['AUC']}")

            logloss = eval_metrics["Logloss"]
            auc = eval_metrics["AUC"]
            for idx in range(len(eval_metrics["AUC"])):
                log_metric("Logloss_val", logloss[idx], step=idx + 1)
                log_metric("Roc_auc_val", auc[idx], step=idx + 1)

            params = model.get_params()
            log_params(params)
            signature = infer_signature(X, model.predict(X))
            mlflow.sklearn.log_model(model, "model", signature=signature)

    elif mode == "infer":
        string = "tags.mlflow.runName = 'TrainRun'"
        run_id = mlflow.search_runs(filter_string=string)["run_id"][0]
        with mlflow.start_run(run_name="TrainRun", run_id=run_id):
            with mlflow.start_run(nested=True, run_name="TestRun"):
                roc_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
                log.info(f"Roc_auc_test: {roc_auc}")
                log_metric("Roc_auc_test", roc_auc)
                params = model.get_params()
                log_params(params)
                signature = infer_signature(X, model.predict(X))
                mlflow.sklearn.log_model(model, "model", signature=signature)

    preds = pd.Series(model.predict(X))
    return preds


def save_components(model, imputer_cat, imputer_num, mode, preds):
    """
    func save model, imputers and predictions for test set
    :param model: trained model
    :param imputer_cat: Imputer for categorical features
    :param imputer_num: Imputer for numerical features
    :param mode: function operating mode
    :param preds: predictions for test set
    :return: None
    """

    if mode == "train":
        model.save_model("../models/cb_clf")

        with open("../models/categorical_imputer.pickle", "wb") as handle:
            pickle.dump(imputer_cat, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("../models/numerical_imputer.pickle", "wb") as handle:
            pickle.dump(imputer_num, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif mode == "infer":
        os.makedirs("../predictions", exist_ok=True)
        preds.to_csv("../predictions/test_preds.csv", index=0)
