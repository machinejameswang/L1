import streamlit as st
import os
import sys

# Add parent dir to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.datasets import generate_dataset
from utils.svm_model import train_svm
from utils.plotting import plot_decision_boundary
from utils.explanations import explain_c_parameter, explain_gamma_parameter, explain_kernel, explain_support_vectors

st.set_page_config(page_title="Interactive SVM", page_icon="🎛️", layout="wide")

st.title("第三章：Interactive SVM 互動實驗室")
st.markdown("在這裡親自動手調整參數，看看決策邊界如何改變！")

# --- Sidebar Controls ---
st.sidebar.header("SVM 參數設定")

dataset_name = st.sidebar.selectbox("1. 選擇資料集 (Dataset)", ["Linearly separable", "Blobs", "Moons", "Circles"])
n_samples = st.sidebar.slider("資料點數量 (n_samples)", 100, 500, 300, step=50)
noise = st.sidebar.slider("雜訊程度 (noise)", 0.0, 0.5, 0.1, step=0.05)

st.sidebar.markdown("---")
kernel_choice = st.sidebar.selectbox("2. 選擇 Kernel", ["linear", "rbf", "poly"])

# C is typically set on a log scale
c_val = st.sidebar.select_slider("3. Regularization (C)", options=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0], value=1.0)

gamma_val = "scale"
if kernel_choice in ["rbf", "poly"]:
    gamma_val = st.sidebar.select_slider("4. Gamma", options=[0.001, 0.01, 0.1, "scale", 1.0, 5.0, 10.0], value="scale")

degree_val = 3
if kernel_choice == "poly":
    degree_val = st.sidebar.slider("5. Polynomial Degree", 2, 5, 3)

# --- Data Generation & Modeling ---
X, y = generate_dataset(dataset_name, n_samples=n_samples, noise=noise)
model, accuracy = train_svm(X, y, kernel=kernel_choice, C=c_val, gamma=gamma_val, degree=degree_val)

svc = model.named_steps['svc']
n_support_vectors = len(svc.support_)

# --- Layout: Plot and Explanations ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Decision Boundary ({dataset_name})")
    fig = plot_decision_boundary(X, y, model, kernel_name=kernel_choice)
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics display
    m1, m2, m3 = st.columns(3)
    m1.metric("Training Accuracy", f"{accuracy:.2%}")
    m2.metric("Number of Support Vectors", f"{n_support_vectors} / {n_samples}")
    m3.metric("Support Vector %", f"{(n_support_vectors/n_samples):.1%}")

with col2:
    st.subheader("參數解讀")
    
    with st.expander("什麼是 Kernel？", expanded=(kernel_choice=="linear")):
        st.markdown(explain_kernel())
        
    with st.expander("參數 C 的影響", expanded=True):
        st.markdown(explain_c_parameter())
        if c_val >= 50.0:
            st.warning("⚠️ 目前 C 值很高！模型可能會過於複雜，導致 Overfitting。")
        elif c_val <= 0.01:
            st.info("ℹ️ 目前 C 值很低，Margin 很寬，但可能容忍過多錯誤 (Underfitting)。")
            
    if kernel_choice in ["rbf", "poly"]:
        with st.expander("參數 Gamma 的影響", expanded=True):
            st.markdown(explain_gamma_parameter())
            if isinstance(gamma_val, float) and gamma_val >= 5.0:
                st.warning("⚠️ 目前 Gamma 值很大，邊界會嚴重包圍單一資料點 (Overfitting)。")

    with st.expander("關於 Support Vectors", expanded=False):
        st.markdown(explain_support_vectors())
        if (n_support_vectors/n_samples) > 0.5:
            st.warning("⚠️ 超過 50% 的資料點都是 Support Vector！這表示模型很難找到乾淨的邊界，或者 C 值設定導致 Margin 包含了大量資料點。")
