"""
model_training.py
-----------------
Trains multiple regression models and returns results for comparison.

n_jobs behaviour
----------------
Default is 1 to ensure reliable execution on Windows (joblib loky
backend can deadlock on some Windows configurations when n_jobs > 1).
Set the environment variable CAR_PRICE_N_JOBS to a higher value on
Linux / macOS machines that support parallel workers, e.g.:

    export CAR_PRICE_N_JOBS=4   # bash / zsh
    $env:CAR_PRICE_N_JOBS = "4" # PowerShell
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Read worker count from environment; default to 1 for Windows safety
_N_JOBS = int(os.environ.get("CAR_PRICE_N_JOBS", "1"))


def prepare_features(df: pd.DataFrame):
    """
    Select model features and target (log_selling_price).

    Drops columns that would cause data leakage or are not usable as
    raw model inputs.

    Returns
    -------
    X            : feature DataFrame
    y            : target Series (log_selling_price)
    feature_names: list of feature column names (same order as X)
    """
    drop_cols = [
        "car_name",           # free-text identifier, not encodable here
        "year",               # replaced by vehicle_age
        "selling_price",      # raw target — must not be a feature
        "depreciation_ratio", # derived from selling_price → leakage
        "log_selling_price",  # IS the target
    ]
    y = df["log_selling_price"].copy()
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    logger.info("Features selected: %s", X.columns.tolist())
    return X, y, X.columns.tolist()


def split_data(X, y, test_size: float = 0.2):
    """Stratification-free train/test split with fixed random state."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    logger.info("Train: %d rows  |  Test: %d rows", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def build_models() -> dict:
    """
    Return {model_name: sklearn Pipeline(StandardScaler + estimator)}.

    Pipelines guarantee the scaler is fit on training data only, preventing
    any form of data leakage through scaling.
    """
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]),
        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01, max_iter=10_000,
                            random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeRegressor(max_depth=8,
                                            random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=_N_JOBS,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
            )),
        ]),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                verbosity=0,
                n_jobs=_N_JOBS,
            )),
        ])
    return models


def train_all_models(X_train, y_train) -> dict:
    """Fit every model in build_models() and return trained pipelines."""
    models = build_models()
    trained: dict = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline
        logger.info("Trained: %s", name)
    return trained


def tune_best_model(X_train, y_train) -> Pipeline:
    """
    Tune a Random Forest with RandomizedSearchCV (5-fold CV, 30 iterations).

    Parameter choices
    -----------------
    n_estimators       : more trees reduce variance; 500 is the upper bound
                         to keep runtime reasonable on small data
    max_depth          : None allows fully grown trees (good for RF);
                         shallow depths act as regularisation
    min_samples_split  : controls minimum node population for a split
    min_samples_leaf   : ensures leaves have enough samples; prevents overfit
    max_features       : 'sqrt' is the RF default; None uses all features

    Returns the best estimator pipeline as chosen by CV R2.
    """
    param_dist = {
        "model__n_estimators":     [100, 200, 300, 500],
        "model__max_depth":        [None, 5, 10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf":  [1, 2, 4],
        "model__max_features":     ["sqrt", "log2", None],
    }
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=_N_JOBS,
        )),
    ])
    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_dist,
        n_iter=30,
        cv=5,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=_N_JOBS,
        verbose=1,
    )
    search.fit(X_train, y_train)
    logger.info("Best CV R2: %.4f", search.best_score_)
    logger.info("Best params: %s", search.best_params_)
    return search.best_estimator_


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from src.data_preprocessing import clean_pipeline
    from src.feature_engineering import feature_engineering_pipeline

    df = clean_pipeline("../data/car_data.csv")
    df = feature_engineering_pipeline(df)
    X, y, feature_names = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    trained_models = train_all_models(X_train, y_train)
    print("Models trained:", list(trained_models.keys()))
