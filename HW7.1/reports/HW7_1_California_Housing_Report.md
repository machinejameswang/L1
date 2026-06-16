# HW7.1 California Housing CRISP-DM Report

## 1. Business Understanding

The goal is to predict California block-group median house value (`median_house_value`) from housing, demographic, location, income, and ocean proximity features.

## 2. Data Understanding

- Original dataset shape: 20,640 rows x 10 columns
- Augmented dataset shape: 30,000 rows x 10 columns
- Augmented CSV: `HW7_CaliforniaHousing_Augmented_30000.csv`
- Augmented Excel: `HW7_CaliforniaHousing_Augmented_30000.xlsx`
- Target mean: $206,374.37
- Target median: $179,400.00
- Missing values are concentrated in `total_bedrooms`.

## 3. Data Preparation

- Median imputation for numeric features
- Most-frequent imputation plus One-Hot Encoding for `ocean_proximity`
- Robust scaling for numeric features
- Engineered features: `rooms_per_household`, `bedrooms_per_room`, `population_per_household`
- Data augmentation: bootstrap sampling with bounded numeric jitter, increasing rows from 20,640 to 30,000

## 4. Modeling Result

Best cross-validation model: **ExtraTrees**

- CV R2 mean: 0.8467
- CV RMSE mean: $45,245.73
- Holdout R2: 0.8537
- Holdout MAE: $28,644.53
- Holdout RMSE: $43,599.85

## 5. Feature Selection Summary

Top voted transformed features:

```
                  feature  corr_vote  mi_vote  rfe_vote  lasso_vote  rf_vote  total_votes
                 latitude          1        1         1           1        1            5
            median_income          1        1         1           1        1            5
   ocean_proximity_INLAND          1        1         1           1        1            5
        bedrooms_per_room          1        1         0           1        1            4
                longitude          0        1         1           1        1            4
ocean_proximity_<1H OCEAN          1        1         1           0        0            3
      rooms_per_household          1        1         0           0        1            3
               households          0        0         1           1        0            2
       housing_median_age          0        0         0           1        1            2
               population          0        0         1           1        0            2
```

## 6. Deployment Simulation

Sample prediction for the first holdout record: **$239,626.21**

## 7. Generated Charts

- `charts/target_distribution.png`
- `charts/geographic_price_scatter.png`
- `charts/correlation_heatmap.png`
- `charts/ocean_proximity_boxplot.png`
- `charts/feature_selection_votes.png`
- `model_comparison_r2.png`
- `permutation_importance.png`
