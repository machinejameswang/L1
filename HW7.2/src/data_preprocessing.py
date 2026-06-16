import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .utils import CATEGORICAL_COLUMNS, CLASS_TARGET, REGRESSION_TARGET


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
    blocked = set(CATEGORICAL_COLUMNS + [CLASS_TARGET, REGRESSION_TARGET])
    return [c for c in X.columns if c not in blocked]


def add_price_category(df):
    out = df.copy()
    q33 = out[REGRESSION_TARGET].quantile(0.33)
    q66 = out[REGRESSION_TARGET].quantile(0.66)
    out[CLASS_TARGET] = pd.cut(
        out[REGRESSION_TARGET],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)
    return out, {"q33": float(q33), "q66": float(q66)}


def augment_classification_data(df, target_rows=30000, random_state=42):
    if len(df) >= target_rows:
        return df.copy()

    rng = np.random.default_rng(random_state)
    extra_count = target_rows - len(df)
    extra = df.sample(extra_count, replace=True, random_state=random_state).reset_index(drop=True)

    jitter_specs = {
        "longitude": 0.015,
        "latitude": 0.015,
        "housing_median_age": 1.5,
        "total_rooms": 0.06,
        "total_bedrooms": 0.06,
        "population": 0.06,
        "households": 0.06,
        "median_income": 0.04,
        REGRESSION_TARGET: 0.035,
    }
    for col, scale in jitter_specs.items():
        if col in ["longitude", "latitude", "housing_median_age"]:
            noise = rng.normal(0, scale, size=extra_count)
        else:
            base = extra[col].fillna(df[col].median()).abs().clip(lower=1)
            noise = rng.normal(0, base * scale, size=extra_count)
        extra[col] = extra[col] + noise

    extra["longitude"] = extra["longitude"].clip(df["longitude"].min(), df["longitude"].max())
    extra["latitude"] = extra["latitude"].clip(df["latitude"].min(), df["latitude"].max())
    extra["housing_median_age"] = extra["housing_median_age"].clip(1, 52).round(0)
    for col in ["total_rooms", "total_bedrooms", "population", "households"]:
        extra[col] = extra[col].clip(lower=1).round(0)
    extra["median_income"] = extra["median_income"].clip(0.4999, 15.0001)
    extra[REGRESSION_TARGET] = extra[REGRESSION_TARGET].clip(14999, 500001).round(0)

    augmented = pd.concat([df, extra], ignore_index=True)
    return augmented.sample(frac=1, random_state=random_state).reset_index(drop=True)


def missing_value_report(df):
    return pd.DataFrame(
        {"missing_count": df.isna().sum(), "missing_rate": df.isna().mean()}
    ).sort_values("missing_count", ascending=False)


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

