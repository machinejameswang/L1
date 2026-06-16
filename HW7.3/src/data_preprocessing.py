import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .utils import CLASS_TARGET, REGRESSION_TARGET


def add_price_class(df):
    out = df.copy()
    q33 = out[REGRESSION_TARGET].quantile(0.33)
    q66 = out[REGRESSION_TARGET].quantile(0.66)
    out[CLASS_TARGET] = pd.cut(
        out[REGRESSION_TARGET],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)
    return out, {"q33": float(q33), "q66": float(q66)}


def augment_boston_housing(df, target_rows=1000, random_state=42):
    if len(df) >= target_rows:
        return df.copy()

    rng = np.random.default_rng(random_state)
    extra_count = target_rows - len(df)
    extra = df.sample(extra_count, replace=True, random_state=random_state).reset_index(drop=True)
    numeric_cols = [c for c in extra.columns if c != "CHAS"]

    for col in numeric_cols:
        base = extra[col].fillna(df[col].median()).abs().clip(lower=1)
        noise = rng.normal(0, base * 0.035, size=extra_count)
        extra[col] = extra[col] + noise

    for col in extra.columns:
        if col == "CHAS":
            extra[col] = extra[col].round(0).clip(0, 1)
        elif col == REGRESSION_TARGET:
            extra[col] = extra[col].clip(5, 50).round(1)
        else:
            lower = 0 if col not in ["B"] else 0
            upper = None
            if col == "AGE":
                upper = 100
            elif col == "NOX":
                upper = 1
            elif col == "PTRATIO":
                upper = 30
            elif col == "B":
                upper = 396.9
            extra[col] = extra[col].clip(lower=lower, upper=upper)

    for col in ["RAD", "TAX"]:
        extra[col] = extra[col].round(0)

    augmented = pd.concat([df, extra], ignore_index=True)
    return augmented.sample(frac=1, random_state=random_state).reset_index(drop=True)


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
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

