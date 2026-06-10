"""
data_preprocessing.py
---------------------
Loads, cleans, and validates the raw car price dataset.

Cleaning steps applied (in order)
----------------------------------
1. Remove duplicate rows
2. Standardise column names  (lowercase, underscores)
3. Coerce numerical columns to correct dtypes
4. Impute missing values      (median for numeric, mode for categorical)
5. Cap outliers in price columns using the 1.5 × IQR Winsorisation rule
"""

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual cleaning functions
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV dataset from *filepath*."""
    try:
        df = pd.read_csv(filepath)
        logger.info("Loaded %d rows × %d columns from '%s'",
                    df.shape[0], df.shape[1], filepath)
        return df
    except FileNotFoundError:
        logger.error("File not found: %s", filepath)
        raise


def inspect_data(df: pd.DataFrame) -> None:
    """Print a structured inspection summary (shape, dtypes, missing, etc.)."""
    sep = "=" * 60
    print(sep)
    print("DATASET INSPECTION SUMMARY")
    print(sep)
    print(f"\nShape          : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn Names   : {df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nDuplicate Rows : {df.duplicated().sum()}")
    print(f"\nNumerical Summary:\n{df.describe()}")
    for col in df.select_dtypes(include="object").columns:
        print(f"\n{col} value counts:\n{df[col].value_counts()}")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and reset the index."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Removed %d duplicate row(s). Remaining: %d",
                n_before - len(df), len(df))
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names and replace spaces with underscores."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    logger.info("Standardised column names: %s", df.columns.tolist())
    return df


def correct_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known numerical columns to numeric dtype."""
    numeric_cols = ["selling_price", "present_price", "driven_kms",
                    "year", "owner"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info("Data types corrected.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in-place:
    - Numerical columns → median  (robust to outliers)
    - Categorical columns → mode
    """
    for col in df.select_dtypes(include=np.number).columns:
        n_miss = int(df[col].isnull().sum())
        if n_miss:
            val = df[col].median()
            df[col] = df[col].fillna(val)
            logger.info("Filled %d missing in '%s' with median=%.4f",
                        n_miss, col, val)

    for col in df.select_dtypes(include="object").columns:
        n_miss = int(df[col].isnull().sum())
        if n_miss:
            val = df[col].mode()[0]
            df[col] = df[col].fillna(val)
            logger.info("Filled %d missing in '%s' with mode='%s'",
                        n_miss, col, val)
    return df


def treat_outliers(
    df: pd.DataFrame,
    columns: list,
    method: str = "cap",
) -> pd.DataFrame:
    """
    Handle outliers using the 1.5 × IQR rule.

    Parameters
    ----------
    columns : list of numerical column names to inspect
    method  : 'cap'  — Winsorise (clip) values to [Q1−1.5·IQR, Q3+1.5·IQR]
              'drop' — remove rows that contain outliers in any listed column
    """
    for col in columns:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        if n_out == 0:
            continue
        if method == "cap":
            df[col] = df[col].clip(lower=lower, upper=upper)
            logger.info(
                "Capped %d outlier(s) in '%s' to [%.2f, %.2f]",
                n_out, col, lower, upper,
            )
        elif method == "drop":
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            logger.info("Dropped %d outlier row(s) in '%s'", n_out, col)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def clean_pipeline(filepath: str) -> pd.DataFrame:
    """
    Run the full cleaning sequence and return a clean DataFrame.

    Suitable as input for feature_engineering_pipeline().
    """
    df = load_data(filepath)
    df = remove_duplicates(df)
    df = standardize_column_names(df)
    df = correct_data_types(df)
    df = handle_missing_values(df)
    # driven_kms is kept raw; it is log-transformed in feature engineering
    df = treat_outliers(df, columns=["selling_price", "present_price"],
                        method="cap")
    logger.info("Cleaning pipeline complete. Final shape: %s", df.shape)
    return df


if __name__ == "__main__":
    df = clean_pipeline("../data/car_data.csv")
    print(df.head())
