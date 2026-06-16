import streamlit as st
from sklearn.model_selection import train_test_split

from src.data_preprocessing import (
    build_preprocessing_pipeline,
    iqr_outlier_report,
    missing_value_report,
    skewness_report,
    transformed_feature_frame,
)
from src.feature_selection import feature_selection_voting
from src.modeling import evaluate_models_cv
from src.utils import TARGET, load_data, split_xy


st.set_page_config(page_title="HW7.1 California Housing", layout="wide")

st.title("HW7.1 - California Housing Price Prediction")
st.caption("CRISP-DM + Feature Engineering + Feature Selection + Regression Modeling")

df = load_data()
st.subheader("Dataset Overview")
st.write("Shape:", df.shape)
st.dataframe(df.head())

tab1, tab2, tab3 = st.tabs(["Data Understanding", "Feature Selection", "Model Benchmark"])

with tab1:
    st.subheader("Missing Value Report")
    st.dataframe(missing_value_report(df))

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(include="all").T)

    st.subheader("Skewness Report")
    st.dataframe(skewness_report(df))

    st.subheader("IQR Outlier Report")
    st.dataframe(iqr_outlier_report(df))

    st.subheader(f"Correlation with {TARGET}")
    corr = df.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    st.dataframe(corr.to_frame("correlation"))

with tab2:
    X, y = split_xy(df)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    transformed = transformed_feature_frame(build_preprocessing_pipeline(), X_train)
    votes, details = feature_selection_voting(transformed, y_train, top_k=8)
    st.subheader("Feature Selection Voting")
    st.dataframe(votes)

    for name, table in details.items():
        st.markdown(f"### {name}")
        st.dataframe(table)

with tab3:
    X, y = split_xy(df)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    results = evaluate_models_cv(X_train, y_train, build_preprocessing_pipeline(), cv=5)
    st.subheader("Model Comparison")
    st.dataframe(results)

