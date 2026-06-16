from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"
TMP_DIR = PROJECT_ROOT / "tmp"

DATA_FILE = DATA_DIR / "housing.csv"
REGRESSION_TARGET = "median_house_value"
CLASS_TARGET = "price_category"
CATEGORICAL_COLUMNS = ["ocean_proximity"]


def ensure_dirs():
    for path in [DATA_DIR, OUTPUT_DIR, OUTPUT_DIR / "charts", REPORT_DIR, TMP_DIR / "matplotlib"]:
        path.mkdir(parents=True, exist_ok=True)


def load_data(path=None):
    path = Path(path) if path else DATA_FILE
    return pd.read_csv(path)


def split_xy(df, target=CLASS_TARGET):
    drop_cols = [target, REGRESSION_TARGET]
    return df.drop(columns=[c for c in drop_cols if c in df.columns]), df[target]

