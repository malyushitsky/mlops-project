import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from utils import (
    calc_metrics_with_logging,
    get_data,
    preprocess_data,
    save_components,
    train_model,
)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    url = "https://github.com/malyushitsky/mlops_prj"
    file = "data/train.csv"
    df = get_data(file, url)
    df, imputer_cat, imputer_num, cat_cols = preprocess_data(df, "train")

    X = df.drop(["Survived"], axis=1)
    y = df["Survived"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cb_params = OmegaConf.to_container(cfg["params"])
    model = train_model(cb_params, X_train, X_val, y_train, y_val, cat_cols)

    calc_metrics_with_logging(model, X_val, y_val, "train")

    save_components(model, imputer_cat, imputer_num, "train", None)


if __name__ == "__main__":
    main()
