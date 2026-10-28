"""Compare baseline regression and classification models on the supplied datasets."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils import INPUT_DIR, make_preprocessor


def compare_car_models():
    path = INPUT_DIR / "car_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    features = ["Make", "Year", "Kilometer", "Fuel Type",
                "Transmission", "Owner", "Seller Type"]
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=features + ["Price"])
    X, y = df[features], df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }
    rows = []
    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor(X_train)),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        rows.append({
            "Model": name,
            "R2": round(r2_score(y_test, pred), 4),
            "MAE": round(mean_absolute_error(y_test, pred), 2),
        })
    print("\nCAR PRICE MODEL COMPARISON")
    print(pd.DataFrame(rows).to_string(index=False))


def compare_titanic_models():
    path = INPUT_DIR / "Titanic_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    df = df.dropna(subset=["Survived"])
    X, y = df[features], df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }
    rows = []
    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor(X_train)),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, pred), 4),
        })
    print("\nTITANIC CLASSIFICATION MODEL COMPARISON")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    compare_car_models()
    compare_titanic_models()
