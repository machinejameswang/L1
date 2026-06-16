import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_data, ensure_dirs, OUTPUT_DIR, REPORT_DIR, split_xy
from src.data_preprocessing import (
    missing_value_report,
    skewness_report,
    iqr_outlier_report,
    build_preprocessing_pipeline
)
from src.feature_selection import feature_selection_voting
from src.modeling import evaluate_models_cv, train_best_model, permutation_importance_report

def main():
    ensure_dirs()

    df = load_data()
    print("Dataset shape:", df.shape)

    # Data Understanding
    missing = missing_value_report(df)
    skew = skewness_report(df)
    outliers = iqr_outlier_report(df)

    missing.to_csv(OUTPUT_DIR / "missing_value_report.csv")
    skew.to_csv(OUTPUT_DIR / "skewness_report.csv")
    outliers.to_csv(OUTPUT_DIR / "iqr_outlier_report.csv", index=False)

    # Version A: keep all features
    X, y = split_xy(df, drop_sensitive=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    feature_pipeline = build_preprocessing_pipeline(log_transform=True, scaler_method="robust")
    X_train_transformed = feature_pipeline.fit_transform(X_train)

    votes, details = feature_selection_voting(X_train_transformed, y_train, top_k=6)
    votes.to_csv(OUTPUT_DIR / "feature_selection_voting.csv", index=False)
    for name, table in details.items():
        table.to_csv(OUTPUT_DIR / f"feature_selection_{name}.csv", index=False)

    model_pipeline = build_preprocessing_pipeline(log_transform=True, scaler_method="robust")
    model_results = evaluate_models_cv(X_train, y_train, preprocessor=model_pipeline, cv=5)
    model_results.to_csv(OUTPUT_DIR / "model_comparison_all_features.csv", index=False)

    best_model_name = model_results.iloc[0]["model"]
    best_model_pipeline = build_preprocessing_pipeline(log_transform=True, scaler_method="robust")
    model, metrics = train_best_model(
        X_train,
        y_train,
        X_test,
        y_test,
        preprocessor=best_model_pipeline,
        model_name=best_model_name,
    )

    permutation = permutation_importance_report(model, X_test, y_test)
    permutation.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False)

    # Version B: remove sensitive B
    X_b, y_b = split_xy(df, drop_sensitive=True)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b, y_b, test_size=0.2, random_state=42)
    pipeline_b = build_preprocessing_pipeline(log_transform=True, scaler_method="robust")
    
    model_results_no_b = evaluate_models_cv(X_train_b, y_train_b, preprocessor=pipeline_b, cv=5)
    model_results_no_b.to_csv(OUTPUT_DIR / "model_comparison_remove_B.csv", index=False)

    summary = {
        "dataset_shape": df.shape,
        "best_model_all_features": best_model_name,
        "holdout_metrics": metrics,
        "top_features_by_vote": votes.head(8).to_dict(orient="records"),
        "best_model_table": model_results.head(5).to_dict(orient="records")
    }
    pd.DataFrame([summary]).to_json(OUTPUT_DIR / "summary.json", force_ascii=False, indent=2)

    print("\nTop Feature Selection Votes:")
    print(votes.head(10))
    print("\nModel Comparison:")
    print(model_results)
    print("\nBest Model:", best_model_name)
    print("Holdout Metrics:", metrics)
    print("\nDone. Outputs saved to:", OUTPUT_DIR)
