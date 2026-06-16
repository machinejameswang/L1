import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.base import clone

from sklearn.pipeline import make_pipeline

def build_model_pipeline(preprocessor, model):
    if preprocessor is None:
        return model
    if isinstance(preprocessor, Pipeline):
        return Pipeline(preprocessor.steps + [("model", model)])
    return make_pipeline(clone(preprocessor), model)

def get_models(random_state=42):
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state)
    }

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            objective="reg:squarederror"
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=random_state,
            verbose=-1
        )
    except Exception:
        pass

    return models

def evaluate_models_cv(X, y, preprocessor=None, cv=5, random_state=42):
    models = get_models(random_state=random_state)
    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows = []
    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error"
    }
    for name, model in models.items():
        pipeline = build_model_pipeline(preprocessor, model)

        scores = cross_validate(pipeline, X, y, cv=kfold, scoring=scoring, n_jobs=-1)
        rows.append({
            "model": name,
            "R2_mean": scores["test_r2"].mean(),
            "R2_std": scores["test_r2"].std(),
            "MAE_mean": -scores["test_neg_mae"].mean(),
            "RMSE_mean": -scores["test_neg_rmse"].mean()
        })
    return pd.DataFrame(rows).sort_values("R2_mean", ascending=False)

def train_best_model(X_train, y_train, X_test, y_test, preprocessor=None, model_name="GradientBoosting", random_state=42):
    models = get_models(random_state=random_state)
    model = models.get(model_name, models["GradientBoosting"])
    
    pipeline = build_model_pipeline(preprocessor, model)
        
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred)
    }
    
    # Return pipeline as model, so permutation importance works correctly
    return pipeline, metrics

def permutation_importance_report(model, X_test, y_test, random_state=42):
    result = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=random_state, n_jobs=-1)
    return pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)
