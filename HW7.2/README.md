# HW7.2 - California Housing Classification Problem

This project converts California Housing from a regression problem into a classification problem.

## Classification Target

The original continuous target `median_house_value` is converted into `price_category`:

- `low`: below the 33rd percentile
- `medium`: between the 33rd and 66th percentiles
- `high`: above the 66th percentile

## Dataset

- Source file: `data/housing.csv`
- Original rows: 20,640
- Augmented rows: 30,000
- Target: `price_category`

## Workflow

1. Data understanding
2. Data augmentation
3. Feature engineering
4. Numeric scaling and categorical one-hot encoding
5. Classification model benchmarking
6. Best model training
7. Confusion matrix, ROC curve, PR curve, and feature importance

## Run

```bash
python main.py
```

Outputs are written to `outputs/`; the report is written to `reports/`.

