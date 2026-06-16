# Data Understanding Report

## Dataset Overview
- Dataset: Boston Housing Dataset
- Target: MEDV
- Problem type: Regression
- Original sample size: 506
- Augmented sample size: 1000

## Data Dictionary

| Feature | Meaning |
|---|---|
| CRIM | Per capita crime rate |
| ZN | Residential land ratio |
| INDUS | Industrial area ratio |
| CHAS | Charles River dummy variable |
| NOX | Nitric oxides concentration |
| RM | Average number of rooms |
| AGE | Older building proportion |
| DIS | Distance to employment centers |
| RAD | Highway accessibility |
| TAX | Property tax rate |
| PTRATIO | Pupil-teacher ratio |
| B | Ethically sensitive racial demographic variable |
| LSTAT | Lower status population percentage |
| MEDV | Median house value |

## Key Data Understanding Findings
1. Missing values should be treated with median imputation.
2. CRIM, ZN, DIS, RAD, and TAX may show skewness or outliers.
3. RM is expected to have positive correlation with MEDV.
4. LSTAT is expected to have negative correlation with MEDV.
5. B should be discussed as an ethical and fairness risk.

## Recommended Data Preparation
- Median imputation
- Log transform: CRIM, ZN, DIS
- RobustScaler for linear models
- IQR and Isolation Forest for outlier analysis
