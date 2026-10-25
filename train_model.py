"""Train and compare a few beginner-friendly house price models."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "train.csv"
KAGGLE_TEST_PATH = ROOT / "data" / "test.csv"
MODEL_PATH = ROOT / "artifacts" / "house_price_model.joblib"
RESULTS_PATH = ROOT / "reports" / "model_results.csv"
INFO_PATH = ROOT / "artifacts" / "model_info.json"
TREND_PATH = ROOT / "reports" / "price_trend.csv"
SUBMISSION_PATH = ROOT / "reports" / "test_predictions.csv"

FEATURES = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt", "LotArea", "Neighborhood"]
NUMERIC = FEATURES[:-1]
CATEGORICAL = ["Neighborhood"]
TARGET = "SalePrice"


def build_preprocessor(scale=False):
    number_steps = [("fill_missing", SimpleImputer(strategy="median"))]
    if scale:
        number_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(number_steps)
    category_pipe = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numbers", numeric_pipe, NUMERIC),
        ("neighborhood", category_pipe, CATEGORICAL),
    ])


def score_model(name, model, x_test, y_test):
    prediction = model.predict(x_test)
    return {
        "Model": name,
        "MAE_USD": round(mean_absolute_error(y_test, prediction), 2),
        "RMSE_USD": round(np.sqrt(mean_squared_error(y_test, prediction)), 2),
        "R2": round(r2_score(y_test, prediction), 4),
    }


def main():
    data = pd.read_csv(DATA_PATH)
    x = data[FEATURES]
    y = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )

    models = {
        "Median price baseline": Pipeline([
            ("prepare", build_preprocessor()),
            ("model", DummyRegressor(strategy="median")),
        ]),
        "Linear regression": Pipeline([
            ("prepare", build_preprocessor(scale=True)),
            ("model", LinearRegression()),
        ]),
        "Random forest": Pipeline([
            ("prepare", build_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=150, max_depth=12, min_samples_leaf=2,
                random_state=42, n_jobs=-1
            )),
        ]),
    }

    results = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        results.append(score_model(name, model, x_test, y_test))

    results_df = pd.DataFrame(results).sort_values("MAE_USD")
    results_df.to_csv(RESULTS_PATH, index=False)
    best_name = results_df.iloc[0]["Model"]

    # Save a fresh copy of the best model, trained on all available rows.
    final_model = models[best_name]
    final_model.fit(x, y)
    joblib.dump(final_model, MODEL_PATH)

    # Kaggle's separate test.csv has no SalePrice labels, so it cannot be used
    # to calculate accuracy. We can still create predictions for those rows.
    competition_test = pd.read_csv(KAGGLE_TEST_PATH)
    test_predictions = pd.DataFrame({
        "Id": competition_test["Id"],
        "SalePrice": final_model.predict(competition_test[FEATURES]),
    })
    test_predictions.to_csv(SUBMISSION_PATH, index=False)

    # A transparent historical reference, grouped by neighborhood and sale year.
    trend = data.groupby(["Neighborhood", "YrSold"], as_index=False)[TARGET].median()
    trend.rename(columns={"YrSold": "Year", TARGET: "MedianSalePrice"}).to_csv(TREND_PATH, index=False)

    info = {
        "dataset": "Supplied Kaggle House Prices: Advanced Regression Techniques files (Ames, Iowa)",
        "rows": int(len(data)),
        "features": FEATURES,
        "target": TARGET,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "split": "80/20 random holdout, random_state=42",
        "selected_model": best_name,
        "selected_model_test_mae_usd": float(results_df.loc[results_df["Model"] == best_name, "MAE_USD"].iloc[0]),
        "currency": "USD",
        "kaggle_test_rows": int(len(competition_test)),
        "note": "The comparison metrics use a held-out split from train.csv. The Kaggle test.csv has no SalePrice labels, so it is used only to create predictions. The final saved model is refit on all labeled rows for the demo website.",
    }
    INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(results_df.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved Kaggle test predictions: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
