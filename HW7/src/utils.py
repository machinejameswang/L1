from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"

TARGET = "MEDV"

FEATURE_DESCRIPTIONS = {
    "CRIM": "Per capita crime rate by town",
    "ZN": "Proportion of residential land zoned for lots over 25,000 sq.ft.",
    "INDUS": "Proportion of non-retail business acres per town",
    "CHAS": "Charles River dummy variable",
    "NOX": "Nitric oxides concentration",
    "RM": "Average number of rooms per dwelling",
    "AGE": "Proportion of owner-occupied units built prior to 1940",
    "DIS": "Weighted distances to Boston employment centers",
    "RAD": "Index of accessibility to radial highways",
    "TAX": "Full-value property-tax rate",
    "PTRATIO": "Pupil-teacher ratio by town",
    "B": "Ethically sensitive racial demographic variable",
    "LSTAT": "Lower status population percentage",
    "MEDV": "Median value of owner-occupied homes"
}

def load_data(path=None):
    if path is None:
        path = DATA_DIR / "HW7_BostonHousing_Augmented_1000.csv"
        if not path.exists():
            path = DATA_DIR / "HousingData.csv"
    return pd.read_csv(path)

def split_xy(df, target=TARGET, drop_sensitive=False):
    data = df.copy()
    if drop_sensitive and "B" in data.columns:
        data = data.drop(columns=["B"])
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
