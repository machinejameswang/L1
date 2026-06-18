import streamlit as st

st.set_page_config(
    page_title="SVM 互動教學網",
    page_icon="🎓",
    layout="wide"
)

st.title("Support Vector Machine (SVM) 互動式教學")
st.markdown("### 歡迎來到 SVM 學習平台！")

st.markdown("""
這是一個專為初學者設計的 SVM 互動教學網站。在這裡，我們將透過視覺化與互動參數調整，帶你一步步了解 Support Vector Machine 的奧秘。

請從左側導覽列選擇你想學習的單元：

1. **SVM 概念**：直覺了解什麼是 SVM 以及它如何尋找最佳分隔線。
2. **Margin 與 Support Vectors**：了解數學原理，以及哪些資料點真正決定了邊界。
3. **Interactive SVM**：親自動手調整 C、gamma、kernel，即時觀看真實 decision boundary 變化。
4. **Kernel Trick**：看看 SVM 如何將非線性資料投影到高維度來進行分類。
5. **小測驗**：測試你對 SVM 的了解程度！

---
**小提醒**：
本網站內建概念教學影片。如果影片無法播放，可能是開發者尚未將本機渲染的 Manim 影片上傳至 `assets/videos/`，但這不會影響您使用互動式工具學習！
""")

# 可選：在這裡加一些簡單的美化樣式
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)
