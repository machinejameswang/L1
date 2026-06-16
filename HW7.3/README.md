# HW7.3 - Boston Housing Classification Problem

This project converts the HW7 Boston Housing regression task into a classification problem.

## Classification Target

The original continuous target `MEDV` is converted into `PRICE_CLASS`:

- `low`: below the 33rd percentile
- `medium`: between the 33rd and 66th percentiles
- `high`: above the 66th percentile

## Dataset

- Source file: `data/HousingData.csv`
- Original rows: 506
- Augmented rows: 1,000
- Target: `PRICE_CLASS`

## Workflow

1. Data understanding
2. Missing value report
3. Data augmentation
4. Median imputation and robust scaling
5. Classification model benchmarking
6. Best model training
7. Confusion matrix, ROC curve, PR curve, feature importance, and report

## Run

```bash
python main.py
```

Outputs are written to `outputs/`; the final report is written to `reports/`.

