"""
feature_engineering.py
-----------------------
Derives new predictive features from the cleaned car dataset.

Features created
----------------
vehicle_age        int    Current year minus manufacturing year.
                          Strong negative price correlate — older = cheaper.
kms_per_year       float  driven_kms / vehicle_age (clipped ≥ 1).
                          Captures intensity of use rather than raw mileage.
brand_popularity   float  Fraction of dataset listings for this car model.
                          Proxy for market demand / resale liquidity.
log_driven_kms     float  log1p(driven_kms).
                          Removes right skew; more linear relationship with
                          log(price).
log_selling_price  float  log1p(selling_price).  ← MODEL TARGET
                          Normalises the highly right-skewed price
                          distribution.

depreciation_ratio float  selling_price / present_price.
                          Kept for EDA but EXCLUDED from model features
                          because it is derived from the target variable
                          (data leakage).

Categorical encoding
--------------------
fuel_type     → one-hot (drop_first=True) → fuel_type_Diesel, fuel_type_Petrol
selling_type  → one-hot (drop_first=True) → selling_type_Individual
transmission  → binary  (1 = Manual, 0 = Automatic)
"""

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CURRENT_YEAR: int = 2024


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def add_vehicle_age(df: pd.DataFrame) -> pd.DataFrame:
    df["vehicle_age"] = CURRENT_YEAR - df["year"]
    logger.info("Added 'vehicle_age'.")
    return df


def add_depreciation_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """EDA feature only — excluded from model to avoid target leakage."""
    df["depreciation_ratio"] = df["selling_price"] / (df["present_price"] + 1e-9)
    logger.info("Added 'depreciation_ratio' (EDA only — excluded from model).")
    return df


def add_kms_per_year(df: pd.DataFrame) -> pd.DataFrame:
    df["kms_per_year"] = df["driven_kms"] / df["vehicle_age"].clip(lower=1)
    logger.info("Added 'kms_per_year'.")
    return df


def add_brand_popularity(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["car_name"].value_counts()
    df["brand_popularity"] = df["car_name"].map(counts) / len(df)
    logger.info("Added 'brand_popularity'.")
    return df


def add_log_selling_price(df: pd.DataFrame) -> pd.DataFrame:
    df["log_selling_price"] = np.log1p(df["selling_price"])
    logger.info("Added 'log_selling_price' (model target).")
    return df


def add_log_driven_kms(df: pd.DataFrame) -> pd.DataFrame:
    df["log_driven_kms"] = np.log1p(df["driven_kms"])
    logger.info("Added 'log_driven_kms'.")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode fuel_type and selling_type (drop_first avoids
    multicollinearity).  Transmission is binary-encoded (1 = Manual).
    """
    df = pd.get_dummies(df, columns=["fuel_type", "selling_type"],
                        drop_first=True)
    df["transmission_encoded"] = (
        df["transmission"].str.strip().str.lower() == "manual"
    ).astype(int)
    df = df.drop(columns=["transmission"])
    logger.info("Categorical encoding complete.")
    return df


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature transformations in the correct order.

    Expects a cleaned DataFrame from data_preprocessing.clean_pipeline().
    Returns a DataFrame ready for prepare_features() in model_training.
    """
    df = add_vehicle_age(df)
    df = add_depreciation_ratio(df)
    df = add_kms_per_year(df)
    df = add_brand_popularity(df)
    df = add_log_selling_price(df)
    df = add_log_driven_kms(df)
    df = encode_categoricals(df)
    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from src.data_preprocessing import clean_pipeline

    df = clean_pipeline("../data/car_data.csv")
    df = feature_engineering_pipeline(df)
    print(df.dtypes)
    print(df.head(3))
