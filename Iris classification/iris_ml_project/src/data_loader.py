"""
data_loader.py
--------------
Handles all data ingestion, validation, and preprocessing steps.
Produces clean train/test splits ready for model training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (
    DATA_PATH, FEATURES, TARGET, DROP_COLS,
    TEST_SIZE, RANDOM_STATE, SPECIES_SHORT
)


# ── Public API ────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """Load raw CSV and perform basic cleaning."""
    df = pd.read_csv(DATA_PATH)
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Normalise species labels to canonical form
    df[TARGET] = df[TARGET].str.strip()

    # Validate expected columns
    missing = [f for f in FEATURES + [TARGET] if f not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return a structured summary dict for reporting."""
    return {
        "n_samples":      len(df),
        "n_features":     len(FEATURES),
        "n_classes":      df[TARGET].nunique(),
        "class_counts":   df[TARGET].value_counts().to_dict(),
        "missing_values": int(df.isnull().sum().sum()),
        "describe":       df[FEATURES].describe().round(3).to_dict(),
    }


def build_splits(df: pd.DataFrame):
    """
    Encode labels, scale features, and return stratified train/test splits.

    Returns
    -------
    X_train_s, X_test_s  : scaled feature arrays
    y_train, y_test       : integer-encoded label arrays
    le                    : fitted LabelEncoder
    scaler                : fitted StandardScaler
    df                    : the cleaned DataFrame (unchanged)
    """
    X = df[FEATURES].values
    le = LabelEncoder()
    y  = le.fit_transform(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, le, scaler
