# PredictIQ — Full-Stack Machine Learning App

PredictIQ is a demonstration project that provides two machine-learning predictions:
- **Used-car price estimation** using a regression pipeline.
- **Titanic passenger survival** using a classification pipeline.

The application uses scikit-learn pipelines, joblib for model persistence, FastAPI for prediction endpoints, and an HTML/CSS/JavaScript frontend.

## Project structure

```text
ml-app/
├── input/
│   ├── car_data.csv
│   └── Titanic_train.csv
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── model_selection.py
│   ├── tune_model.py
│   └── utils.py
├── models/
│   ├── car_price_model.pkl
│   └── titanic_model.pkl
├── notebooks/
│   └── exploration.ipynb
├── static/
│   └── index.html
├── app.py
├── requirements.txt
└── README.md
```

## Requirements

Use Python 3.10 or another Python version compatible with the installed dependencies. Activate your virtual environment before installing packages.

```powershell
python -m pip install -r requirements.txt
```

If `requirements.txt` does not yet exist, create one containing:

```text
fastapi
uvicorn[standard]
pandas
scikit-learn
joblib
```

## Dataset files

Place the datasets in the `input` folder with these exact names:

- `car_data.csv` — must contain `Price` and the features `Make`, `Year`, `Kilometer`, `Fuel Type`, `Transmission`, `Owner`, and `Seller Type`.
- `Titanic_train.csv` — must contain `Survived`, `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.

The scripts expect these exact file names and column names.

## Train both models

From the project root:

```powershell
python src/train.py
```

The training script should create:
- `models/car_price_model.pkl`
- `models/titanic_model.pkl`

It prints the regression R² and mean absolute error, plus Titanic classification accuracy.

## Run the web application

From the project root:

```powershell
python -m uvicorn app:app --reload
```

Open:
- Frontend: http://127.0.0.1:8000/
- API documentation: http://127.0.0.1:8000/docs

## Use the prediction CLI

Car price example:

```powershell
python src/predict.py car --make Honda --year 2019 --kilometers 50000 --fuel-type Petrol --transmission Manual --owner First --seller-type Individual
```

Titanic example:

```powershell
python src/predict.py titanic --pclass 3 --sex male --age 22 --sibsp 1 --parch 0 --fare 7.25 --embarked S
```

## Compare models

```powershell
python src/model_selection.py
```

This compares Linear Regression with Random Forest for car prices and Logistic Regression with Random Forest for Titanic survival. It prints R²/MAE for regression and accuracy for classification.

## Tune the car model

```powershell
python src/tune_model.py
```

This runs randomized cross-validation for a Random Forest regressor and saves the tuned pipeline to `models/car_price_model_tuned.pkl`. It does **not** overwrite the model currently used by `app.py`.

## Explore the data

Open `notebooks/exploration.ipynb` in VS Code or Jupyter and run the cells in order. The notebook loads both datasets and shows dimensions, column types, missing values, summary statistics, and basic target distributions.

## Notes

- Predictions are estimates, not guarantees of actual market price or historical outcomes.
- Model scores depend on the supplied data and the train/test split.
- Do not commit `.venv/`, large datasets, or sensitive files to GitHub unless your submission requires them.
