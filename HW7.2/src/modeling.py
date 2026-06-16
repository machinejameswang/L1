import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_model_pipeline(preprocessor, model):
    return Pipeline(preprocessor.steps + [("model", clone(model))])


def get_models(random_state=42):
    return {
        "LogisticRegression": LogisticRegression(max_iter=5000, class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced", random_state=random_state, max_iter=10000),
        "RandomForest": RandomForestClassifier(n_estimators=120, random_state=random_state, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=120, random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
    }


def evaluate_models_cv(X, y, preprocessor, cv=5, random_state=42):
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows = []
    for name, model in get_models(random_state).items():
        pipeline = build_model_pipeline(preprocessor, model)
        scores = cross_validate(pipeline, X, y, cv=kfold, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "model": name,
                "accuracy_mean": scores["test_accuracy"].mean(),
                "f1_macro_mean": scores["test_f1_macro"].mean(),
                "precision_macro_mean": scores["test_precision_macro"].mean(),
                "recall_macro_mean": scores["test_recall_macro"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)


def train_best_model(X_train, y_train, X_test, y_test, preprocessor, model_name, random_state=42):
    model = get_models(random_state)[model_name]
    pipeline = build_model_pipeline(preprocessor, model)
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
    }
    return pipeline, pred, metrics


def save_model(model, path):
    joblib.dump(model, path)

