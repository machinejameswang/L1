import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest

LOG_COLUMNS = ["CRIM", "ZN", "DIS"]

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

LOG_COLUMNS = ["CRIM", "ZN", "DIS"]

def missing_value_report(df):
    report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_rate": df.isna().mean()
    })
    return report.sort_values("missing_count", ascending=False)

def skewness_report(df):
    numeric = df.select_dtypes(include=[np.number])
    return numeric.skew().sort_values(ascending=False).to_frame("skewness")

def iqr_outlier_report(df):
    numeric = df.select_dtypes(include=[np.number])
    rows = []
    for col in numeric.columns:
        q1 = numeric[col].quantile(0.25)
        q3 = numeric[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = ((numeric[col] < lower) | (numeric[col] > upper)).sum()
        rows.append({
            "feature": col,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(count),
            "outlier_rate": float(count / len(df))
        })
    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)

# DataFrame preserving Custom Transformers
class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="median"):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.columns = None
    def fit(self, X, y=None):
        self.columns = X.columns
        self.imputer.fit(X)
        return self
    def transform(self, X):
        return pd.DataFrame(self.imputer.transform(X), columns=self.columns, index=X.index)

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=LOG_COLUMNS):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        out = X.copy()
        for col in self.cols:
            if col in out.columns:
                out[col] = np.log1p(out[col].clip(lower=0))
        return out

class DataFrameScaler(BaseEstimator, TransformerMixin):
    def __init__(self, method="robust"):
        self.method = method
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = RobustScaler()
        self.columns = None
    def fit(self, X, y=None):
        self.columns = X.columns
        self.scaler.fit(X)
        return self
    def transform(self, X):
        return pd.DataFrame(self.scaler.transform(X), columns=self.columns, index=X.index)

def build_preprocessing_pipeline(log_transform=True, scaler_method="robust"):
    steps = [
        ("imputer", DataFrameImputer(strategy="median"))
    ]
    if log_transform:
        steps.append(("log_transform", LogTransformer()))
    steps.append(("scaler", DataFrameScaler(method=scaler_method)))
    
    return Pipeline(steps)

