
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score


# -------- PROJECT PATHS --------

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -------- PREPROCESSING FUNCTION --------

def make_preprocessor(X):
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])


# ==================================================
# MODEL 1: CAR PRICE REGRESSION
# ==================================================

print("\nLoading car dataset...")

car_path = INPUT_DIR / "car_data.csv"
car_df = pd.read_csv(car_path)

car_features = [
    "Make",
    "Year",
    "Kilometer",
    "Fuel Type",
    "Transmission",
    "Owner",
    "Seller Type"
]

target_column = "Price"

# Check required columns
required_car_columns = car_features + [target_column]
missing_columns = [
    col for col in required_car_columns
    if col not in car_df.columns
]

if missing_columns:
    raise ValueError(
        f"Car dataset is missing these columns: {missing_columns}"
    )

# Ensure target is numeric
car_df[target_column] = pd.to_numeric(
    car_df[target_column], errors="coerce"
)

# Remove rows with missing target
car_df = car_df.dropna(subset=[target_column]).copy()

# Remove rows with missing input features
car_df = car_df.dropna(subset=car_features).copy()

if len(car_df) < 10:
    raise ValueError(
        "Not enough valid rows in the car dataset to train the model."
    )

X_car = car_df[car_features]
y_car = car_df[target_column]

X_car_train, X_car_test, y_car_train, y_car_test = train_test_split(
    X_car,
    y_car,
    test_size=0.2,
    random_state=42
)

car_model = Pipeline([
    ("preprocessor", make_preprocessor(X_car_train)),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

print("Training car price model...")
car_model.fit(X_car_train, y_car_train)

car_predictions = car_model.predict(X_car_test)

car_r2 = r2_score(y_car_test, car_predictions)
car_mae = mean_absolute_error(y_car_test, car_predictions)

print("\n--- CAR PRICE REGRESSION RESULTS ---")
print(f"R² Score: {car_r2:.4f}")
print(f"Mean Absolute Error: {car_mae:.2f}")

joblib.dump(car_model, MODEL_DIR / "car_price_model.pkl")
print("Saved: models/car_price_model.pkl")


# ==================================================
# MODEL 2: TITANIC SURVIVAL CLASSIFICATION
# ==================================================

print("\nLoading Titanic dataset...")

titanic_path = INPUT_DIR / "Titanic_train.csv"
titanic = pd.read_csv(titanic_path)

titanic_features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

titanic_target = "Survived"

# Check required columns
required_titanic_columns = titanic_features + [titanic_target]
missing_columns = [
    col for col in required_titanic_columns
    if col not in titanic.columns
]

if missing_columns:
    raise ValueError(
        f"Titanic dataset is missing these columns: {missing_columns}"
    )

# Remove rows without a survival label
titanic = titanic.dropna(subset=[titanic_target]).copy()

X_titanic = titanic[titanic_features]
y_titanic = titanic[titanic_target]

if y_titanic.nunique() != 2:
    raise ValueError(
        "The Survived column must contain two classes: 0 and 1."
    )

X_titanic_train, X_titanic_test, y_titanic_train, y_titanic_test = (
    train_test_split(
        X_titanic,
        y_titanic,
        test_size=0.2,
        random_state=42,
        stratify=y_titanic
    )
)

titanic_model = Pipeline([
    ("preprocessor", make_preprocessor(X_titanic_train)),
    ("model", LogisticRegression(max_iter=1000))
])

print("Training Titanic classification model...")
titanic_model.fit(X_titanic_train, y_titanic_train)

titanic_predictions = titanic_model.predict(X_titanic_test)

titanic_accuracy = accuracy_score(
    y_titanic_test,
    titanic_predictions
)

print("\n--- TITANIC CLASSIFICATION RESULTS ---")
print(f"Accuracy: {titanic_accuracy:.4f}")

joblib.dump(titanic_model, MODEL_DIR / "titanic_model.pkl")
print("Saved: models/titanic_model.pkl")


# -------- COMPLETION --------

print("\nBoth models trained and saved successfully!")
