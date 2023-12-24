import io
import pickle

import dvc.api
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer


def main():
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

    model = CatBoostClassifier(train_dir=None, allow_writing_files=False)

    model.fit(X, y, verbose=False, plot=False, cat_features=cat_cols)

    model.save_model("models/cb_clf")

    with open("models/categorical_imputer.pickle", "wb") as handle:
        pickle.dump(imputer_cat, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("models/numerical_imputer.pickle", "wb") as handle:
        pickle.dump(imputer_num, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
