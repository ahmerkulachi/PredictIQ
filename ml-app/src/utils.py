"""Reusable preprocessing and project-path helpers for PredictIQ."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
MODEL_DIR = ROOT / "models"


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing based on numeric and categorical input columns."""
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def load_car_data() -> pd.DataFrame:
    """Load the car dataset expected at input/car_data.csv."""
    path = INPUT_DIR / "car_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Car dataset not found: {path}")
    return pd.read_csv(path)


def load_titanic_data() -> pd.DataFrame:
    """Load the Titanic dataset expected at input/Titanic_train.csv."""
    path = INPUT_DIR / "Titanic_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Titanic dataset not found: {path}")
    return pd.read_csv(path)
