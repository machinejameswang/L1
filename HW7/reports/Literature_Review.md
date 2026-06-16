# Literature Review - Boston Housing Dataset

## 1. Dataset Background
The Boston Housing Dataset originates from Harrison and Rubinfeld's 1978 hedonic pricing study. It contains 506 observations and 14 variables, including the target variable `MEDV`.

## 2. Important Features from Literature and Kaggle Practice
Commonly important features include:

| Feature | Reason |
|---|---|
| RM | Average number of rooms; usually positively correlated with house value |
| LSTAT | Lower status population percentage; usually negatively correlated with house value |
| PTRATIO | Pupil-teacher ratio; represents education-related neighborhood quality |
| DIS | Distance to employment centers; represents accessibility |
| NOX | Air pollution indicator |
| CRIM | Crime rate |
| TAX | Property tax rate |

## 3. Recommended Algorithms
Boston Housing is a small tabular regression dataset with nonlinear relationships. Recommended algorithms:

1. Gradient Boosting
2. Random Forest
3. Extra Trees
4. XGBoost
5. LightGBM
6. Ridge / Lasso / ElasticNet as interpretable baselines

## 4. Ethical Consideration
The `B` variable is ethically sensitive because it encodes racial demographic information. A professional project should compare two versions:

- Version A: all features
- Version B: remove `B`

If performance loss is small, the recommended deployment version should remove `B`.

## 5. HW7 Research Hypotheses
- H1: RM and LSTAT are the most important predictors.
- H2: Tree-based ensemble models outperform linear models.
- H3: Log transformation improves skewed features such as CRIM, ZN, and DIS.
- H4: Removing `B` may reduce ethical risk with limited performance loss.
