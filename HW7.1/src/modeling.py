import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline


def build_model_pipeline(preprocessor, model):
    return Pipeline(preprocessor.steps + [("model", clone(model))])


def get_models(random_state=42):
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0),
        "Lasso": Lasso(alpha=100.0, max_iter=20000),
        "ElasticNet": ElasticNet(alpha=100.0, l1_ratio=0.5, max_iter=20000),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
    }


def evaluate_models_cv(X, y, preprocessor, cv=5, random_state=42):
    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
    }
    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows = []
    for name, model in get_models(random_state).items():
        pipeline = build_model_pipeline(preprocessor, model)
        scores = cross_validate(pipeline, X, y, cv=kfold, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "model": name,
                "R2_mean": scores["test_r2"].mean(),
                "R2_std": scores["test_r2"].std(),
                "MAE_mean": -scores["test_neg_mae"].mean(),
                "RMSE_mean": -scores["test_neg_rmse"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("R2_mean", ascending=False)


def train_best_model(X_train, y_train, X_test, y_test, preprocessor, model_name, random_state=42):
    model = get_models(random_state)[model_name]
    pipeline = build_model_pipeline(preprocessor, model)
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred),
    }
    return pipeline, metrics


def permutation_importance_report(model, X_test, y_test, random_state=42):
    result = permutation_importance(
        model, X_test, y_test, n_repeats=8, random_state=random_state, n_jobs=-1
    )
    return pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def save_model(model, path):
    joblib.dump(model, path)

