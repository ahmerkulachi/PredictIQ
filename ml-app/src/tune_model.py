"""Tune a Random Forest car-price model with randomized cross-validation search."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from utils import INPUT_DIR, MODEL_DIR, make_preprocessor


def main():
    data_path = INPUT_DIR / "car_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    features = ["Make", "Year", "Kilometer", "Fuel Type",
                "Transmission", "Owner", "Seller Type"]
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=features + ["Price"])

    X, y = df[features], df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", make_preprocessor(X_train)),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])

    parameter_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_grid,
        n_iter=12,
        scoring="neg_mean_absolute_error",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    print("Tuning Random Forest. This may take a few minutes...")
    search.fit(X_train, y_train)

    predictions = search.best_estimator_.predict(X_test)
    print("\nBest parameters:", search.best_params_)
    print(f"Test R²: {r2_score(y_test, predictions):.4f}")
    print(f"Test MAE: {mean_absolute_error(y_test, predictions):.2f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_DIR / "car_price_model_tuned.pkl"
    joblib.dump(search.best_estimator_, output_path)
    print(f"Tuned model saved to: {output_path}")


if __name__ == "__main__":
    main()
