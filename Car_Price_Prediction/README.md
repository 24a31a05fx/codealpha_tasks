# Car Price Prediction

Predict used car resale prices with a clean, reproducible machine learning pipeline.

## Project Overview

This project trains regression models to estimate the selling price of a used car from attributes such as showroom price, kilometers driven, fuel type, transmission, seller type, owner count, and vehicle age.

The pipeline includes data cleaning, exploratory plots, feature engineering, model comparison, Random Forest tuning, and model artifact saving.

## Project Structure

```text
Car_Price_Prediction/
+-- data/
|   +-- car_data.csv
+-- models/
|   +-- best_model.pkl
|   +-- feature_names.pkl
|   +-- tuned_random_forest.pkl
+-- notebooks/
|   +-- Car_Price_Prediction.ipynb
+-- outputs/
|   +-- plots/
|   +-- reports/
|       +-- model_comparison.csv
+-- src/
|   +-- __init__.py
|   +-- data_preprocessing.py
|   +-- feature_engineering.py
|   +-- model_evaluation.py
|   +-- model_training.py
+-- .gitignore
+-- main.py
+-- README.md
+-- requirements.txt
```

## Dataset

| Column | Description |
| --- | --- |
| `Car_Name` | Vehicle model name |
| `Year` | Manufacturing year |
| `Selling_Price` | Target resale price in lakhs INR |
| `Present_Price` | Current showroom price in lakhs INR |
| `Driven_kms` | Total kilometers driven |
| `Fuel_Type` | Petrol, Diesel, or CNG |
| `Selling_type` | Dealer or Individual |
| `Transmission` | Manual or Automatic |
| `Owner` | Number of previous owners |

After cleaning, the dataset has 299 rows, no missing values, and no duplicate rows.

## Installation

```bash
pip install -r requirements.txt
```

XGBoost is optional. If it is installed, the training module can include it automatically; it is not required for the default pipeline.

## Run the Pipeline

```bash
python main.py
```

The run creates or updates:

- `outputs/plots/` for EDA and diagnostic charts
- `outputs/reports/model_comparison.csv` for model metrics
- `models/best_model.pkl` for the best held-out test model
- `models/tuned_random_forest.pkl` for the tuned Random Forest comparison model
- `models/feature_names.pkl` for the feature order used during training

By default, model training uses one worker for reliable Windows execution. To enable more workers on a machine that supports it:

```bash
$env:CAR_PRICE_N_JOBS = "4"
python main.py
```

## Current Results

Metrics are computed on the log-transformed selling price target.

| Model | R2 | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Ridge Regression | 0.9242 | 0.1603 | 0.2143 |
| Linear Regression | 0.9238 | 0.1610 | 0.2148 |
| Lasso Regression | 0.9169 | 0.1666 | 0.2243 |
| Random Forest | 0.9077 | 0.1369 | 0.2364 |
| Gradient Boosting | 0.9075 | 0.1413 | 0.2366 |
| Decision Tree | 0.8437 | 0.1667 | 0.3076 |

`best_model.pkl` currently saves Ridge Regression because it has the highest held-out test R2.

## Use the Saved Model

The saved model expects engineered features in the same order as `feature_names.pkl`.

```python
import joblib
import numpy as np

model = joblib.load("models/best_model.pkl")
features = joblib.load("models/feature_names.pkl")

sample = np.array([[
    9.85,      # present_price
    27000,     # driven_kms
    0,         # owner
    12,        # vehicle_age
    2250.0,    # kms_per_year
    0.0334,    # brand_popularity
    10.2036,   # log_driven_kms
    0,         # fuel_type_Diesel
    1,         # fuel_type_Petrol
    0,         # selling_type_Individual
    1,         # transmission_encoded, 1 = Manual
]])

log_price = model.predict(sample)[0]
price_lakhs = np.expm1(log_price)
print(f"Predicted price: INR {price_lakhs:.2f} lakhs")
```

## Notes

- The project avoids target leakage: engineered training features do not use `selling_price`.
- `vehicle_age` is calculated from the current calendar year.
- This is a small dataset, so linear models can outperform larger ensemble models on the held-out split.
