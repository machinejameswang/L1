# HW7.2 California Housing Classification Report

## 1. Problem Definition

This project turns California Housing into a classification problem by predicting `price_category` from housing, demographic, income, location, and ocean proximity features.

## 2. Target Design

- `low`: `median_house_value` <= $140,192.41
- `medium`: between $140,192.41 and $227,700.00
- `high`: `median_house_value` > $227,700.00

## 3. Dataset

- Original shape: 20,640 rows x 10 columns
- Augmented shape: 30,000 rows x 11 columns
- Augmented CSV: `HW7_CaliforniaHousing_Classification_Augmented_30000.csv`
- Augmented Excel: `HW7_CaliforniaHousing_Classification_Augmented_30000.xlsx`

## 4. Best Model

Best CV model: **ExtraTrees**

- CV macro F1: 0.8473
- Holdout accuracy: 0.8662
- Holdout macro F1: 0.8651
- Holdout macro precision: 0.8649
- Holdout macro recall: 0.8659

## 5. Classification Report

```
              precision    recall  f1-score   support

        high       0.89      0.89      0.89      2040
         low       0.88      0.93      0.91      1980
      medium       0.82      0.78      0.80      1980

    accuracy                           0.87      6000
   macro avg       0.86      0.87      0.87      6000
weighted avg       0.87      0.87      0.87      6000

```

## 6. Generated Charts

- `charts/class_balance.png`
- `charts/value_by_class_boxplot.png`
- `charts/confusion_matrix.png`
- `charts/model_comparison_f1.png`
- `charts/roc_curve_ovr.png`
- `charts/precision_recall_curve.png`
