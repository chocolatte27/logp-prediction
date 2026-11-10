"""Train the final XGBoost model that predicts logP from 8 molecular descriptors."""
import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "MolWt", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "FractionCSP3", "HeavyAtomCount",
]
TARGET = "logP_target"


def load_and_clean(path):
    df = pd.read_csv(path, usecols=["InChIKey"] + FEATURES + [TARGET])
    df = df.drop_duplicates(subset="InChIKey").reset_index(drop=True)
    # Keep only normal drug-like molecules (removes extreme outliers)
    df = df[
        df[TARGET].between(-5, 10)
        & (df["MolWt"] <= 1000)
        & (df["HeavyAtomCount"] <= 80)
        & (df["NumRotatableBonds"] <= 30)
    ].reset_index(drop=True)
    return df[FEATURES + [TARGET]]


def evaluate(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main(data_path, model_path, metrics_path):
    data = load_and_clean(data_path)
    print("Clean rows:", len(data))

    X, y = data[FEATURES], data[TARGET]
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    # Best settings found by hyperparameter tuning in the notebook
    model = xgb.XGBRegressor(
        n_estimators=3000, early_stopping_rounds=50,
        learning_rate=0.1, max_depth=12, min_child_weight=5,
        subsample=0.9, colsample_bytree=1.0, reg_lambda=5,
        n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print("Best number of trees:", model.best_iteration)

    metrics = {
        "validation": evaluate(y_val, model.predict(X_val)),
        "test": evaluate(y_test, model.predict(X_test)),
        "best_iteration": int(model.best_iteration),
        "features": FEATURES,
    }
    print(json.dumps(metrics, indent=2))

    model.save_model(model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved model to", model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to the CSV file")
    parser.add_argument("--model-out", default="model.ubj")
    parser.add_argument("--metrics-out", default="metrics.json")
    args = parser.parse_args()
    main(args.data, args.model_out, args.metrics_out)
