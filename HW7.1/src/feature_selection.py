import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.linear_model import LassoCV, LinearRegression


def correlation_selection(X, y):
    df = X.copy()
    df["target"] = y.values
    corr = df.corr(numeric_only=True)["target"].drop("target")
    return corr.sort_values(key=lambda s: s.abs(), ascending=False).to_frame("correlation")


def mutual_information_selection(X, y, random_state=42):
    mi = mutual_info_regression(X, y, random_state=random_state)
    return pd.DataFrame({"feature": X.columns, "mutual_information": mi}).sort_values(
        "mutual_information", ascending=False
    )


def rfe_selection(X, y, n_features=8):
    selector = RFE(LinearRegression(), n_features_to_select=min(n_features, X.shape[1]))
    selector.fit(X, y)
    return pd.DataFrame(
        {"feature": X.columns, "rfe_selected": selector.support_, "rfe_rank": selector.ranking_}
    ).sort_values(["rfe_selected", "rfe_rank"], ascending=[False, True])


def lasso_selection(X, y, random_state=42):
    model = LassoCV(cv=5, random_state=random_state, max_iter=50000)
    model.fit(X, y)
    return pd.DataFrame(
        {"feature": X.columns, "lasso_coef": model.coef_, "abs_coef": np.abs(model.coef_)}
    ).sort_values("abs_coef", ascending=False)


def random_forest_importance(X, y, random_state=42):
    model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    model.fit(X, y)
    return pd.DataFrame({"feature": X.columns, "rf_importance": model.feature_importances_}).sort_values(
        "rf_importance", ascending=False
    )


def feature_selection_voting(X, y, top_k=8):
    corr = correlation_selection(X, y).reset_index().rename(columns={"index": "feature"})
    mi = mutual_information_selection(X, y)
    rfe = rfe_selection(X, y, n_features=top_k)
    lasso = lasso_selection(X, y)
    rf = random_forest_importance(X, y)

    votes = pd.DataFrame({"feature": X.columns})
    votes["corr_vote"] = votes["feature"].isin(corr.head(top_k)["feature"]).astype(int)
    votes["mi_vote"] = votes["feature"].isin(mi.head(top_k)["feature"]).astype(int)
    votes["rfe_vote"] = votes["feature"].isin(rfe[rfe["rfe_selected"]]["feature"]).astype(int)
    votes["lasso_vote"] = votes["feature"].isin(lasso.head(top_k)["feature"]).astype(int)
    votes["rf_vote"] = votes["feature"].isin(rf.head(top_k)["feature"]).astype(int)
    votes["total_votes"] = votes[[c for c in votes.columns if c.endswith("_vote")]].sum(axis=1)
    return votes.sort_values(["total_votes", "feature"], ascending=[False, True]), {
        "correlation": corr,
        "mutual_information": mi,
        "rfe": rfe,
        "lasso": lasso,
        "random_forest": rf,
    }

