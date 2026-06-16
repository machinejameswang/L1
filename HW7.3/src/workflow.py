import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, classification_report, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from src.data_preprocessing import (
    add_price_class,
    augment_boston_housing,
    build_preprocessing_pipeline,
    iqr_outlier_report,
    missing_value_report,
    skewness_report,
)
from src.modeling import (
    evaluate_models_cv,
    permutation_importance_report,
    save_model,
    train_best_model,
)
from src.utils import CLASS_TARGET, DATA_DIR, OUTPUT_DIR, REGRESSION_TARGET, REPORT_DIR, ensure_dirs, load_data, split_xy


RANDOM_STATE = 42
CLASS_ORDER = ["low", "medium", "high"]


def save_augmented_dataset(df):
    csv_path = DATA_DIR / f"HW7_BostonHousing_Classification_Augmented_{len(df)}.csv"
    xlsx_path = DATA_DIR / f"HW7_BostonHousing_Classification_Augmented_{len(df)}.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path


def predict_scores(model, X_test):
    clf = model.named_steps["model"]
    if hasattr(clf, "predict_proba"):
        return model.predict_proba(X_test)
    scores = model.decision_function(X_test)
    scores = scores - scores.min(axis=1, keepdims=True)
    denom = scores.sum(axis=1, keepdims=True)
    return scores / denom.clip(min=1e-12)


def plot_charts(df, model_results, y_test, y_pred, y_score, classes, permutation):
    chart_dir = OUTPUT_DIR / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=CLASS_TARGET, hue=CLASS_TARGET, order=CLASS_ORDER, palette="Set2", legend=False)
    plt.title("Boston MEDV Price Class Balance")
    plt.tight_layout()
    plt.savefig(chart_dir / "class_balance.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=CLASS_TARGET, y=REGRESSION_TARGET, hue=CLASS_TARGET, order=CLASS_ORDER, palette="Set2", legend=False)
    plt.title("MEDV by Price Class")
    plt.tight_layout()
    plt.savefig(chart_dir / "medv_by_class_boxplot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 8))
    corr = df.drop(columns=[CLASS_TARGET]).corr(numeric_only=True)
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=False, square=True, cbar_kws={"shrink": 0.8})
    plt.title("Boston Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(chart_dir / "correlation_heatmap.png", dpi=160)
    plt.close()

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(chart_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(model_results["model"], model_results["f1_macro_mean"], color="#2563eb")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean CV Macro F1")
    plt.title("Boston Classification Model Comparison")
    plt.tight_layout()
    plt.savefig(chart_dir / "model_comparison_f1.png", dpi=160)
    plt.close()

    y_bin = label_binarize(y_test, classes=classes)
    plt.figure(figsize=(7, 6))
    for idx, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_score[:, idx])
        plt.plot(fpr, tpr, label=f"{class_name} AUC={auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], color="#6b7280", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_dir / "roc_curve_ovr.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 6))
    for idx, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, idx], y_score[:, idx])
        plt.plot(recall, precision, label=class_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("One-vs-Rest Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_dir / "precision_recall_curve.png", dpi=160)
    plt.close()

    top = permutation.head(10).sort_values("importance_mean")
    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["importance_mean"], color="#16a34a")
    plt.xlabel("Permutation Importance")
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(chart_dir / "feature_importance.png", dpi=160)
    plt.close()


def write_report(original_df, df, thresholds, paths, model_results, metrics, report_text):
    best = model_results.iloc[0]
    content = f"""# HW7.3 Boston Housing Classification Report

## 1. Problem Definition

This project converts the HW7 Boston Housing regression task into a classification problem by predicting `{CLASS_TARGET}` from the housing features.

## 2. Target Design

- `low`: `{REGRESSION_TARGET}` <= {thresholds['q33']:.2f}
- `medium`: between {thresholds['q33']:.2f} and {thresholds['q66']:.2f}
- `high`: `{REGRESSION_TARGET}` > {thresholds['q66']:.2f}

## 3. Dataset

- Original shape: {original_df.shape[0]:,} rows x {original_df.shape[1]} columns
- Augmented shape: {df.shape[0]:,} rows x {df.shape[1]} columns
- Augmented CSV: `{paths[0].name}`
- Augmented Excel: `{paths[1].name}`
- Sensitive feature note: `B` is included for HW7 compatibility, but should be reviewed for fairness concerns.

## 4. Best Model

Best CV model: **{best['model']}**

- CV macro F1: {best['f1_macro_mean']:.4f}
- Holdout accuracy: {metrics['accuracy']:.4f}
- Holdout macro F1: {metrics['f1_macro']:.4f}
- Holdout macro precision: {metrics['precision_macro']:.4f}
- Holdout macro recall: {metrics['recall_macro']:.4f}

## 5. Classification Report

```
{report_text}
```

## 6. Generated Charts

- `charts/class_balance.png`
- `charts/medv_by_class_boxplot.png`
- `charts/correlation_heatmap.png`
- `charts/confusion_matrix.png`
- `charts/model_comparison_f1.png`
- `charts/roc_curve_ovr.png`
- `charts/precision_recall_curve.png`
- `charts/feature_importance.png`
"""
    (REPORT_DIR / "HW7_3_Boston_Housing_Classification_Report.md").write_text(content, encoding="utf-8")


def main():
    ensure_dirs()
    original_df = load_data()
    augmented = augment_boston_housing(original_df, target_rows=1000, random_state=RANDOM_STATE)
    df, thresholds = add_price_class(augmented)
    paths = save_augmented_dataset(df)

    missing_value_report(df).to_csv(OUTPUT_DIR / "missing_value_report.csv")
    skewness_report(df).to_csv(OUTPUT_DIR / "skewness_report.csv")
    iqr_outlier_report(df).to_csv(OUTPUT_DIR / "iqr_outlier_report.csv", index=False)
    df[CLASS_TARGET].value_counts().rename_axis(CLASS_TARGET).reset_index(name="count").to_csv(
        OUTPUT_DIR / "class_balance.csv", index=False
    )

    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model_results = evaluate_models_cv(X_train, y_train, build_preprocessing_pipeline(), cv=5)
    model_results.to_csv(OUTPUT_DIR / "classification_model_comparison.csv", index=False)

    best_model_name = model_results.iloc[0]["model"]
    model, y_pred, metrics = train_best_model(
        X_train, y_train, X_test, y_test, build_preprocessing_pipeline(), best_model_name
    )
    save_model(model, OUTPUT_DIR / "boston_housing_classification_model.joblib")

    classes = list(model.classes_)
    y_score = predict_scores(model, X_test)
    report_text = classification_report(y_test, y_pred, labels=classes)
    (OUTPUT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")

    permutation = permutation_importance_report(model, X_test, y_test)
    permutation.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False)
    plot_charts(df, model_results, y_test, y_pred, y_score, classes, permutation)

    summary = {
        "original_dataset_shape": list(original_df.shape),
        "augmented_dataset_shape": list(df.shape),
        "target": CLASS_TARGET,
        "thresholds": thresholds,
        "best_model": best_model_name,
        "holdout_metrics": metrics,
        "class_order": classes,
        "augmented_csv": str(paths[0]),
        "augmented_xlsx": str(paths[1]),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(original_df, df, thresholds, paths, model_results, metrics, report_text)

    print("Original dataset shape:", original_df.shape)
    print("Augmented dataset shape:", df.shape)
    print("Class thresholds:", thresholds)
    print("\nModel comparison:")
    print(model_results)
    print("\nBest model:", best_model_name)
    print("Holdout metrics:", metrics)
    print("\nDone. Outputs saved to:", OUTPUT_DIR)


