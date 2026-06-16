import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.utils import load_data, split_xy
from src.data_preprocessing import missing_value_report, skewness_report, iqr_outlier_report, build_preprocessing_pipeline
from src.feature_selection import feature_selection_voting
from src.modeling import evaluate_models_cv

st.set_page_config(page_title="HW7 Boston Housing ML Project", layout="wide")

st.title("HW7 - AI-Assisted Boston Housing Price Prediction")
st.caption("CRISP-DM + Data Understanding + Feature Selection + Regression Modeling")

df = load_data()
st.subheader("Dataset Overview")
st.write("Shape:", df.shape)
st.dataframe(df.head())

tab1, tab2, tab3, tab4 = st.tabs(["Data Understanding", "Feature Selection", "Model Benchmark", "Ethical Version"])

with tab1:
    st.subheader("Missing Value Report")
    st.dataframe(missing_value_report(df))

    st.subheader("Statistical Summary")
    st.dataframe(df.describe().T)

    st.subheader("Skewness Report")
    st.dataframe(skewness_report(df))

    st.subheader("IQR Outlier Report")
    st.dataframe(iqr_outlier_report(df))

    st.subheader("Correlation with MEDV")
    corr = df.corr(numeric_only=True)["MEDV"].drop("MEDV").sort_values(key=lambda s: s.abs(), ascending=False)
    st.dataframe(corr.to_frame("correlation"))

with tab2:
    X, y = split_xy(df, drop_sensitive=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = build_preprocessing_pipeline()
    X_train_transformed = pipeline.fit_transform(X_train)
    
    votes, details = feature_selection_voting(X_train_transformed, y_train, top_k=6)
    st.subheader("Feature Selection Voting (on Training Set)")
    st.dataframe(votes)

    for name, table in details.items():
        st.markdown(f"### {name}")
        st.dataframe(table)

with tab3:
    X, y = split_xy(df, drop_sensitive=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = build_preprocessing_pipeline()
    
    results = evaluate_models_cv(X_train, y_train, preprocessor=pipeline, cv=5)
    st.subheader("Model Comparison - All Features (CV on Training Set)")
    st.dataframe(results)

with tab4:
    st.subheader("Ethical Comparison: Keep B vs Remove B")
    X1, y1 = split_xy(df, drop_sensitive=False)
    X2, y2 = split_xy(df, drop_sensitive=True)
    
    X1_train, _, y1_train, _ = train_test_split(X1, y1, test_size=0.2, random_state=42)
    X2_train, _, y2_train, _ = train_test_split(X2, y2, test_size=0.2, random_state=42)

    pipe1 = build_preprocessing_pipeline()
    pipe2 = build_preprocessing_pipeline()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Version A: Keep All Features")
        st.dataframe(evaluate_models_cv(X1_train, y1_train, preprocessor=pipe1, cv=5))
    with c2:
        st.markdown("### Version B: Remove B")
        st.dataframe(evaluate_models_cv(X2_train, y2_train, preprocessor=pipe2, cv=5))

st.info("Recommended conclusion: if removing B causes little performance loss, remove B for ethical and fairness reasons.")
