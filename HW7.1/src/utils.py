from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"
TMP_DIR = PROJECT_ROOT / "tmp"

DATA_FILE = DATA_DIR / "housing.csv"
TARGET = "median_house_value"
CATEGORICAL_COLUMNS = ["ocean_proximity"]


def ensure_dirs():
    for path in [DATA_DIR, OUTPUT_DIR, REPORT_DIR, TMP_DIR, TMP_DIR / "matplotlib"]:
        path.mkdir(parents=True, exist_ok=True)


def load_data(path=None):
    path = Path(path) if path else DATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def split_xy(df, target=TARGET):
    return df.drop(columns=[target]), df[target]

