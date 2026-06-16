# HW7.3 Boston Housing Classification Report

## 1. Problem Definition

This project converts the HW7 Boston Housing regression task into a classification problem by predicting `PRICE_CLASS` from the housing features.

## 2. Target Design

- `low`: `MEDV` <= 18.70
- `medium`: between 18.70 and 23.70
- `high`: `MEDV` > 23.70

## 3. Dataset

- Original shape: 506 rows x 14 columns
- Augmented shape: 1,000 rows x 15 columns
- Augmented CSV: `HW7_BostonHousing_Classification_Augmented_1000.csv`
- Augmented Excel: `HW7_BostonHousing_Classification_Augmented_1000.xlsx`
- Sensitive feature note: `B` is included for HW7 compatibility, but should be reviewed for fairness concerns.

## 4. Best Model

Best CV model: **RandomForest**

- CV macro F1: 0.8397
- Holdout accuracy: 0.8550
- Holdout macro F1: 0.8555
- Holdout macro precision: 0.8572
- Holdout macro recall: 0.8548

## 5. Classification Report

```
              precision    recall  f1-score   support

        high       0.90      0.90      0.90        67
         low       0.90      0.85      0.88        67
      medium       0.77      0.82      0.79        66

    accuracy                           0.85       200
   macro avg       0.86      0.85      0.86       200
weighted avg       0.86      0.85      0.86       200

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
