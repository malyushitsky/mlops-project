import io
import logging
import os
import pickle

import dvc.api
import hydra
import pandas as pd
from catboost import CatBoostClassifier
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    url = "https://github.com/malyushitsky/mlops_prj"
    data = dvc.api.read("data/test.csv", repo=url)
    df = pd.read_csv(io.StringIO(data), sep=",")

    with open("models/categorical_imputer.pickle", "rb") as handle:
        imputer_cat = pickle.load(handle)
    with open("models/numerical_imputer.pickle", "rb") as handle:
        imputer_num = pickle.load(handle)

    model = CatBoostClassifier()
    model.load_model("models/cb_clf")

    cat_cols = df.select_dtypes(include=["object"]).columns.to_list()
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.to_list()

    df[cat_cols] = imputer_cat.transform(df[cat_cols])
    df[num_cols] = imputer_num.transform(df[num_cols])

    X = df.drop(["Survived"], axis=1)
    y = df["Survived"]

    roc_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    log.info(f"Roc_auc_test: {roc_auc}")
    preds = pd.Series(model.predict(X))

    os.makedirs("./predictions", exist_ok=True)
    preds.to_csv("predictions/test_preds.csv", index=0)


if __name__ == "__main__":
    main()
