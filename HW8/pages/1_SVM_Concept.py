import streamlit as st
import os

st.set_page_config(page_title="SVM 概念", page_icon="📖")

st.title("第一章：SVM 是什麼？")

st.markdown("""
### 直覺介紹
Support Vector Machine (支持向量機，簡稱 SVM) 是一種強大的機器學習演算法。
在最簡單的二元分類問題中，我們的目標是找到一條「線」（或是高維度空間中的超平面 Hyperplane），把兩種不同類別的資料乾淨地分開。

### 為什麼不是隨便找一條分隔線？
如果兩群資料可以被一條直線分開，那理論上可以畫出無限多條直線。
但哪一條是最好的呢？
SVM 的哲學是：**找到最「寬」的那條路！**
我們不只希望把資料分開，還希望這條線離兩邊的資料點「越遠越好」，這樣未來遇到新資料時，才不容易預測錯誤。

### 核心概念：
- **Hyperplane (超平面)**：用來分隔資料的決策邊界（在 2D 中是一條直線，3D 中是一個平面）。
- **Margin (邊界距離)**：Hyperplane 到最近資料點的距離。SVM 的目標就是最大化這個 Margin！
- **Support Vectors (支持向量)**：真正決定 Margin 寬度的那些邊緣資料點。

---
### 概念動畫展示：最大化 Margin
""")

video_path = "assets/videos/svm_margin_intro.mp4"
if os.path.exists(video_path):
    st.video(video_path)
else:
    st.warning("⚠️ 尚未產生概念教學影片！請參閱 README 使用 Manim 於本機渲染 `svm_margin_scene.py`，並放置於 `assets/videos/svm_margin_intro.mp4`。")
