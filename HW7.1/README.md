# HW7.1 - California Housing Price Prediction

This project follows the HW7 CRISP-DM workflow and rebuilds the regression pipeline for the California Housing dataset.

## Goal

Predict `median_house_value` using geographic, demographic, housing stock, income, and ocean proximity features.

## Dataset

- Source: `https://raw.githubusercontent.com/jaynilpatel/california-housing/refs/heads/master/datasets/housing/housing.csv`
- Original rows: 20,640
- Augmented rows: 30,000
- Target: `median_house_value`
- Categorical feature: `ocean_proximity`
- Missing values: `total_bedrooms`

## Workflow

1. Data understanding
2. Missing value and outlier reports
3. Data augmentation with bootstrap sampling and bounded numeric jitter
4. Feature engineering
5. Numeric scaling and categorical one-hot encoding
6. Feature selection voting
7. Regression model benchmarking
8. Best model training and deployment simulation

## Generated Charts

- `outputs/charts/target_distribution.png`
- `outputs/charts/geographic_price_scatter.png`
- `outputs/charts/correlation_heatmap.png`
- `outputs/charts/ocean_proximity_boxplot.png`
- `outputs/charts/feature_selection_votes.png`
- `outputs/model_comparison_r2.png`
- `outputs/permutation_importance.png`

## Run

```bash
python main.py
```

Dashboard:

```bash
streamlit run streamlit_app.py
```

Outputs are written to `outputs/`, reports to `reports/`, and the final model to `outputs/california_housing_model.joblib`.
