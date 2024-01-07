import hydra
from catboost import CatBoostClassifier
from omegaconf import DictConfig
from utils import calc_metrics_with_logging, get_data, preprocess_data, save_components


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(_: DictConfig):
    url = "https://github.com/malyushitsky/mlops_prj"
    file = "data/test.csv"
    df = get_data(file, url)

    df = preprocess_data(df, "infer")

    X = df.drop(["Survived"], axis=1)
    y = df["Survived"]

    model = CatBoostClassifier()
    model.load_model("../models/cb_clf")

    preds = calc_metrics_with_logging(model, X, y, "infer")

    save_components(None, None, None, "infer", preds)


if __name__ == "__main__":
    main()
