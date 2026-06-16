import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .utils import CATEGORICAL_COLUMNS


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        households = out["households"].replace(0, np.nan)
        rooms = out["total_rooms"].replace(0, np.nan)
        out["rooms_per_household"] = out["total_rooms"] / households
        out["bedrooms_per_room"] = out["total_bedrooms"] / rooms
        out["population_per_household"] = out["population"] / households
        return out.replace([np.inf, -np.inf], np.nan)


def numeric_columns_selector(X):
    return [c for c in X.columns if c not in CATEGORICAL_COLUMNS]


def missing_value_report(df):
    return pd.DataFrame(
        {"missing_count": df.isna().sum(), "missing_rate": df.isna().mean()}
    ).sort_values("missing_count", ascending=False)


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
        rows.append(
            {
                "feature": col,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": int(count),
                "outlier_rate": float(count / len(df)),
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)


def build_preprocessing_pipeline():
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_columns_selector),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("feature_engineering", HousingFeatureEngineer()), ("preprocessor", preprocessor)])


def transformed_feature_frame(preprocessor, X):
    transformed = preprocessor.fit_transform(X)
    names = preprocessor.named_steps["preprocessor"].get_feature_names_out()
    return pd.DataFrame(transformed, columns=names, index=X.index)
