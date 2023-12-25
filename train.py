import io
import logging
import pickle

import dvc.api
import hydra
import pandas as pd
from catboost import CatBoostClassifier
from omegaconf import DictConfig, OmegaConf
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    url = "https://github.com/malyushitsky/mlops_prj"
    data = dvc.api.read("data/train.csv", repo=url)
    df = pd.read_csv(io.StringIO(data), sep=",")

    cat_cols = df.select_dtypes(include=["object"]).columns.to_list()
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.to_list()

    imputer_cat = SimpleImputer(strategy="most_frequent")
    imputer_num = SimpleImputer(strategy="median")
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

    X = df.drop(["Survived"], axis=1)
    y = df["Survived"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = CatBoostClassifier(train_dir=None, allow_writing_files=False)
    cb_params = OmegaConf.to_container(cfg["params"])
    model.set_params(**cb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        verbose=False,
        plot=False,
        use_best_model=True,
        cat_features=cat_cols,
    )

    logloss = log_loss(y_val, model.predict_proba(X_val))
    roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    log.info(f"Logloss_val: {logloss}")
    log.info(f"Roc_auc_val: {roc_auc}")

    model.save_model("models/cb_clf")

    with open("models/categorical_imputer.pickle", "wb") as handle:
        pickle.dump(imputer_cat, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("models/numerical_imputer.pickle", "wb") as handle:
        pickle.dump(imputer_num, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
