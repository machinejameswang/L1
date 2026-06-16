# HW7 - AI-Assisted Boston Housing Price Prediction System

## Project Goal
This HW7 project follows the CRISP-DM framework to solve the Boston Housing regression problem.

The goal is to predict `MEDV`, the median value of owner-occupied homes, using data understanding, data preparation, feature selection, regression modeling, and explainable AI.

## Dataset
- Original dataset: Boston Housing Dataset
- Original samples: 506
- Augmented dataset: 1000 samples
- Target variable: `MEDV`

## CRISP-DM Workflow
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

## Main Technical Skills
- Python
- Pandas / NumPy
- Scikit-Learn
- Feature Selection
- PCA
- Regression Modeling
- XGBoost / LightGBM optional
- SHAP optional
- Streamlit Dashboard

## Recommended Pipeline
```text
Median Imputation
↓
Log Transform: CRIM, ZN, DIS
↓
RobustScaler / StandardScaler
↓
Feature Selection:
    Correlation
    Mutual Information
    RFE
    Lasso
    Random Forest Importance
↓
Regression Benchmark:
    Linear Regression
    Ridge
    Lasso
    ElasticNet
    Random Forest
    Extra Trees
    Gradient Boosting
    XGBoost optional
    LightGBM optional
↓
Explainable AI:
    Permutation Importance
    SHAP optional
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run full analysis
```bash
python main.py
```

### 3. Run Streamlit App
```bash
streamlit run streamlit_app.py
```

## Expected Findings
Important features are expected to include:

```text
RM
LSTAT
PTRATIO
DIS
CRIM
NOX
```

Tree-based models such as Gradient Boosting, Random Forest, XGBoost, and LightGBM are expected to perform better than pure linear models.

## Ethical Note
The `B` feature is ethically sensitive because it encodes racial demographic information. This project recommends comparing two versions:

- Version A: keep all features
- Version B: remove `B`

If model performance difference is small, Version B is preferred for fairness and ethical considerations.
