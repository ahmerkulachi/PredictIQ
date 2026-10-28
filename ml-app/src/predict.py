"""Run predictions from the saved PredictIQ models.

Examples from the project root:
  python src/predict.py car --make Honda --year 2019 --kilometers 50000 --fuel-type Petrol --transmission Manual --owner First --seller-type Individual
  python src/predict.py titanic --pclass 3 --sex male --age 22 --sibsp 1 --parch 0 --fare 7.25 --embarked S
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def load_model(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Run `python src/train.py` first."
        )
    return joblib.load(path)


def predict_car(args):
    model = load_model("car_price_model.pkl")
    row = pd.DataFrame([{
        "Make": args.make,
        "Year": args.year,
        "Kilometer": args.kilometers,
        "Fuel Type": args.fuel_type,
        "Transmission": args.transmission,
        "Owner": args.owner,
        "Seller Type": args.seller_type,
    }])
    prediction = float(model.predict(row)[0])
    print(f"Estimated car price: {prediction:,.2f} (dataset price units)")


def predict_titanic(args):
    model = load_model("titanic_model.pkl")
    row = pd.DataFrame([{
        "Pclass": args.pclass,
        "Sex": args.sex,
        "Age": args.age,
        "SibSp": args.sibsp,
        "Parch": args.parch,
        "Fare": args.fare,
        "Embarked": args.embarked,
    }])
    prediction = int(model.predict(row)[0])
    classes = list(model.classes_)
    probability = model.predict_proba(row)[0][classes.index(1)]
    print("Prediction:", "Survived" if prediction == 1 else "Did not survive")
    print(f"Estimated survival probability: {probability:.1%}")


def main():
    parser = argparse.ArgumentParser(description="PredictIQ command-line predictions")
    subparsers = parser.add_subparsers(dest="task", required=True)

    car = subparsers.add_parser("car", help="Estimate a used-car price")
    car.add_argument("--make", required=True)
    car.add_argument("--year", type=int, required=True)
    car.add_argument("--kilometers", type=float, required=True)
    car.add_argument("--fuel-type", dest="fuel_type", required=True)
    car.add_argument("--transmission", required=True)
    car.add_argument("--owner", required=True)
    car.add_argument("--seller-type", dest="seller_type", required=True)
    car.set_defaults(func=predict_car)

    titanic = subparsers.add_parser("titanic", help="Predict Titanic survival")
    titanic.add_argument("--pclass", type=int, choices=[1, 2, 3], required=True)
    titanic.add_argument("--sex", choices=["male", "female"], required=True)
    titanic.add_argument("--age", type=float, required=True)
    titanic.add_argument("--sibsp", type=int, default=0)
    titanic.add_argument("--parch", type=int, default=0)
    titanic.add_argument("--fare", type=float, required=True)
    titanic.add_argument("--embarked", choices=["S", "C", "Q"], required=True)
    titanic.set_defaults(func=predict_titanic)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
