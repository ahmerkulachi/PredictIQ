
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# ==================================================
# PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
STATIC_DIR = ROOT / "static"

CAR_MODEL_PATH = MODEL_DIR / "car_price_model.pkl"
TITANIC_MODEL_PATH = MODEL_DIR / "titanic_model.pkl"
INDEX_PATH = STATIC_DIR / "index.html"


# ==================================================
# LOAD TRAINED MODELS
# ==================================================

if not CAR_MODEL_PATH.exists():
    raise FileNotFoundError(
        "Car model not found. Run: python src/train.py"
    )

if not TITANIC_MODEL_PATH.exists():
    raise FileNotFoundError(
        "Titanic model not found. Run: python src/train.py"
    )

car_model = joblib.load(CAR_MODEL_PATH)
titanic_model = joblib.load(TITANIC_MODEL_PATH)


# ==================================================
# FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="PredictIQ - Machine Learning Studio",
    description="Predict used-car prices and Titanic passenger survival.",
    version="1.0.0"
)

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static"
    )


# ==================================================
# INPUT SCHEMAS
# ==================================================

class CarPredictionInput(BaseModel):
    make: str = Field(min_length=1)
    year: int = Field(ge=1900, le=2100)
    kilometers: float = Field(ge=0)
    fuel_type: str = Field(min_length=1)
    transmission: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    seller_type: str = Field(min_length=1)


class TitanicPredictionInput(BaseModel):
    Pclass: int = Field(ge=1, le=3)
    Sex: str = Field(min_length=1)
    Age: float = Field(ge=0, le=120)
    SibSp: int = Field(ge=0)
    Parch: int = Field(ge=0)
    Fare: float = Field(ge=0)
    Embarked: str = Field(min_length=1)


# ==================================================
# HOMEPAGE
# ==================================================

@app.get("/")
def home():
    if not INDEX_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend not found. Check static/index.html."
        )

    return FileResponse(INDEX_PATH)


# ==================================================
# ENDPOINT 1: CAR PRICE REGRESSION
# ==================================================

@app.post("/predict/regression")
def predict_car_price(data: CarPredictionInput):
    try:
        input_df = pd.DataFrame([{
            "Make": data.make.strip(),
            "Year": data.year,
            "Kilometer": data.kilometers,
            "Fuel Type": data.fuel_type.strip(),
            "Transmission": data.transmission.strip(),
            "Owner": data.owner.strip(),
            "Seller Type": data.seller_type.strip()
        }])

        prediction = float(car_model.predict(input_df)[0])

        return {
            "predicted_price": round(prediction, 2),
            "message": "Car price predicted successfully."
        }

    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid car input: {error}"
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Car prediction failed. Check the server terminal."
        ) from error


# ==================================================
# ENDPOINT 2: TITANIC CLASSIFICATION
# ==================================================

@app.post("/predict/classification")
def predict_survival(data: TitanicPredictionInput):
    try:
        input_df = pd.DataFrame([data.model_dump()])

        prediction = int(titanic_model.predict(input_df)[0])
        probabilities = titanic_model.predict_proba(input_df)[0]

        # Find the probability corresponding to class 1.
        classes = list(titanic_model.classes_)
        survived_index = classes.index(1)
        survival_probability = float(
            probabilities[survived_index]
        )

        return {
            "survived": prediction,
            "result": (
                "Survived" if prediction == 1
                else "Did not survive"
            ),
            "survival_probability": round(
                survival_probability, 4
            ),
            "message": "Passenger survival predicted successfully."
        }

    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Titanic input: {error}"
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Titanic prediction failed. Check the server terminal."
        ) from error
