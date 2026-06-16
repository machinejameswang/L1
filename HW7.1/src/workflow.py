import json
import os

os.environ.setdefault("MPLCONFIGDIR", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "tmp" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.data_preprocessing import (
    build_preprocessing_pipeline,
    iqr_outlier_report,
    missing_value_report,
    skewness_report,
    transformed_feature_frame,
)
from src.feature_selection import feature_selection_voting
from src.modeling import (
    evaluate_models_cv,
    permutation_importance_report,
    save_model,
    train_best_model,
)
from src.utils import DATA_DIR, OUTPUT_DIR, REPORT_DIR, TARGET, ensure_dirs, load_data, split_xy


RANDOM_STATE = 42
AUGMENTED_ROWS = 30000


def augment_california_housing(df, target_rows=AUGMENTED_ROWS, random_state=RANDOM_STATE):
    """Bootstrap original rows and add small domain-bounded numeric jitter."""
    if len(df) >= target_rows:
        return df.copy()

    rng = __import__("numpy").random.default_rng(random_state)
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
        TARGET: 0.035,
    }
    for col, scale in jitter_specs.items():
        if col in ["longitude", "latitude", "housing_median_age"]:
            noise = rng.normal(0, scale, size=extra_count)
        else:
            base = extra[col].fillna(df[col].median()).abs().clip(lower=1)
            noise = rng.normal(0, base * scale, size=extra_count)
        extra[col] = extra[col] + noise

    bounds = {
        "longitude": (df["longitude"].min(), df["longitude"].max()),
        "latitude": (df["latitude"].min(), df["latitude"].max()),
        "housing_median_age": (1, 52),
        "total_rooms": (1, None),
        "total_bedrooms": (1, None),
        "population": (1, None),
        "households": (1, None),
        "median_income": (0.4999, 15.0001),
        TARGET: (14999, 500001),
    }
    for col, (lower, upper) in bounds.items():
        extra[col] = extra[col].clip(lower=lower, upper=upper)

    for col in ["housing_median_age", "total_rooms", "total_bedrooms", "population", "households", TARGET]:
        extra[col] = extra[col].round(0)

    augmented = pd.concat([df, extra], ignore_index=True)
    return augmented.sample(frac=1, random_state=random_state).reset_index(drop=True)


def save_augmented_dataset(df):
    csv_path = DATA_DIR / f"HW7_CaliforniaHousing_Augmented_{len(df)}.csv"
    xlsx_path = DATA_DIR / f"HW7_CaliforniaHousing_Augmented_{len(df)}.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path


def write_report(original_df, df, model_results, metrics, votes, prediction, augmented_paths):
    best = model_results.iloc[0]
    report = f"""# HW7.1 California Housing CRISP-DM Report

## 1. Business Understanding

The goal is to predict California block-group median house value (`{TARGET}`) from housing, demographic, location, income, and ocean proximity features.

## 2. Data Understanding

- Original dataset shape: {original_df.shape[0]:,} rows x {original_df.shape[1]} columns
- Augmented dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns
- Augmented CSV: `{augmented_paths[0].name}`
- Augmented Excel: `{augmented_paths[1].name}`
- Target mean: ${df[TARGET].mean():,.2f}
- Target median: ${df[TARGET].median():,.2f}
- Missing values are concentrated in `total_bedrooms`.

## 3. Data Preparation

- Median imputation for numeric features
- Most-frequent imputation plus One-Hot Encoding for `ocean_proximity`
- Robust scaling for numeric features
- Engineered features: `rooms_per_household`, `bedrooms_per_room`, `population_per_household`
- Data augmentation: bootstrap sampling with bounded numeric jitter, increasing rows from {original_df.shape[0]:,} to {df.shape[0]:,}

## 4. Modeling Result

Best cross-validation model: **{best['model']}**

- CV R2 mean: {best['R2_mean']:.4f}
- CV RMSE mean: ${best['RMSE_mean']:,.2f}
- Holdout R2: {metrics['R2']:.4f}
- Holdout MAE: ${metrics['MAE']:,.2f}
- Holdout RMSE: ${metrics['RMSE']:,.2f}

## 5. Feature Selection Summary

Top voted transformed features:

```
{votes.head(10).to_string(index=False)}
```

## 6. Deployment Simulation

Sample prediction for the first holdout record: **${prediction:,.2f}**

## 7. Generated Charts

- `charts/target_distribution.png`
- `charts/geographic_price_scatter.png`
- `charts/correlation_heatmap.png`
- `charts/ocean_proximity_boxplot.png`
- `charts/feature_selection_votes.png`
- `model_comparison_r2.png`
- `permutation_importance.png`
"""
    (REPORT_DIR / "HW7_1_California_Housing_Report.md").write_text(report, encoding="utf-8")


def plot_outputs(df, model_results, votes, permutation):
    chart_dir = OUTPUT_DIR / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 5))
    sns.histplot(df[TARGET], bins=45, kde=True, color="#2563eb")
    plt.xlabel("Median House Value")
    plt.ylabel("Count")
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(chart_dir / "target_distribution.png", dpi=160)
    plt.close()

    sample = df.sample(min(8000, len(df)), random_state=RANDOM_STATE)
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        sample["longitude"],
        sample["latitude"],
        c=sample[TARGET],
        s=(sample["population"].clip(1, 5000) / 5000) * 35 + 4,
        cmap="viridis",
        alpha=0.55,
        linewidths=0,
    )
    plt.colorbar(scatter, label="Median House Value")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Geographic Price Scatter")
    plt.tight_layout()
    plt.savefig(chart_dir / "geographic_price_scatter.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(chart_dir / "correlation_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="ocean_proximity", y=TARGET, hue="ocean_proximity", palette="Set2", legend=False)
    plt.xticks(rotation=20, ha="right")
    plt.xlabel("Ocean Proximity")
    plt.ylabel("Median House Value")
    plt.title("House Value by Ocean Proximity")
    plt.tight_layout()
    plt.savefig(chart_dir / "ocean_proximity_boxplot.png", dpi=160)
    plt.close()

    top_votes = votes.head(10).sort_values("total_votes")
    plt.figure(figsize=(9, 5))
    plt.barh(top_votes["feature"], top_votes["total_votes"], color="#7c3aed")
    plt.xlabel("Feature Selection Votes")
    plt.title("Top Feature Selection Votes")
    plt.tight_layout()
    plt.savefig(chart_dir / "feature_selection_votes.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(model_results["model"], model_results["R2_mean"], color="#2563eb")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean CV R2")
    plt.title("California Housing Model Comparison")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison_r2.png", dpi=160)
    plt.close()

    top = permutation.head(8).sort_values("importance_mean")
    plt.figure(figsize=(9, 5))
    plt.barh(top["feature"], top["importance_mean"], color="#16a34a")
    plt.xlabel("Permutation Importance")
    plt.title("Top Raw Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "permutation_importance.png", dpi=160)
    plt.close()


def main():
    ensure_dirs()

    original_df = load_data()
    df = augment_california_housing(original_df)
    augmented_paths = save_augmented_dataset(df)
    print("Original dataset shape:", original_df.shape)
    print("Augmented dataset shape:", df.shape)
    print("Augmented files:", augmented_paths[0], augmented_paths[1])

    missing_value_report(df).to_csv(OUTPUT_DIR / "missing_value_report.csv")
    skewness_report(df).to_csv(OUTPUT_DIR / "skewness_report.csv")
    iqr_outlier_report(df).to_csv(OUTPUT_DIR / "iqr_outlier_report.csv", index=False)

    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    feature_preprocessor = build_preprocessing_pipeline()
    X_train_transformed = transformed_feature_frame(feature_preprocessor, X_train)
    votes, details = feature_selection_voting(X_train_transformed, y_train, top_k=8)
    votes.to_csv(OUTPUT_DIR / "feature_selection_voting.csv", index=False)
    for name, table in details.items():
        table.to_csv(OUTPUT_DIR / f"feature_selection_{name}.csv", index=False)

    model_results = evaluate_models_cv(
        X_train, y_train, preprocessor=build_preprocessing_pipeline(), cv=5
    )
    model_results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    best_model_name = model_results.iloc[0]["model"]
    model, metrics = train_best_model(
        X_train,
        y_train,
        X_test,
        y_test,
        preprocessor=build_preprocessing_pipeline(),
        model_name=best_model_name,
    )
    save_model(model, OUTPUT_DIR / "california_housing_model.joblib")

    permutation = permutation_importance_report(model, X_test, y_test)
    permutation.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False)
    plot_outputs(df, model_results, votes, permutation)

    sample_prediction = float(model.predict(X_test.head(1))[0])
    summary = {
        "dataset_shape": list(df.shape),
        "original_dataset_shape": list(original_df.shape),
        "augmented_csv": str(augmented_paths[0]),
        "augmented_xlsx": str(augmented_paths[1]),
        "target": TARGET,
        "best_model": best_model_name,
        "holdout_metrics": metrics,
        "sample_prediction": sample_prediction,
        "top_features_by_vote": votes.head(10).to_dict(orient="records"),
        "model_table": model_results.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(original_df, df, model_results, metrics, votes, sample_prediction, augmented_paths)

    print("\nTop Feature Selection Votes:")
    print(votes.head(10))
    print("\nModel Comparison:")
    print(model_results)
    print("\nBest Model:", best_model_name)
    print("Holdout Metrics:", metrics)
    print("Sample Prediction:", f"${sample_prediction:,.2f}")
    print("\nDone. Outputs saved to:", OUTPUT_DIR)

