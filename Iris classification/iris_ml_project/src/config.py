"""
config.py
---------
Central configuration for the Iris ML pipeline.
All tuneable parameters live here — edit this file to change behaviour
without touching model or pipeline code.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "iris.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ── Dataset ───────────────────────────────────────────────────────────────────
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET   = "Species"
DROP_COLS = ["Id"]

FEATURE_DISPLAY = {
    "SepalLengthCm": "Sepal Length (cm)",
    "SepalWidthCm":  "Sepal Width (cm)",
    "PetalLengthCm": "Petal Length (cm)",
    "PetalWidthCm":  "Petal Width (cm)",
}

SPECIES_SHORT = {
    "Iris-setosa":     "setosa",
    "Iris-versicolor": "versicolor",
    "Iris-virginica":  "virginica",
}

# ── Experiment ────────────────────────────────────────────────────────────────
TEST_SIZE    = 0.25
RANDOM_STATE = 42
CV_FOLDS     = 5

# ── Visualisation ─────────────────────────────────────────────────────────────
PALETTE     = ["#6C63FF", "#FF6584", "#43BBAD"]   # setosa / versicolor / virginica
COLOR_BG    = "#0f172a"
COLOR_CARD  = "#1e293b"
COLOR_BORDER= "#334155"
COLOR_MUTED = "#94a3b8"
COLOR_BEST  = "#43BBAD"
COLOR_BASE  = "#6C63FF"
COLOR_ACCENT= "#FF6584"
PLOT_DPI    = 130
