import streamlit as st
import os

st.set_page_config(page_title="Margin 與 Support Vectors", page_icon="📐")

st.title("第二章：Margin 與 Support Vectors")

st.markdown(r"""
在上一章，我們學到 SVM 的目標是尋找「最寬的決策邊界 (Maximum Margin)」。現在我們來看看數學上這是怎麼做到的。

### 決策函數與超平面
假設我們的資料可以用一條線完美分開，這條線的數學式可以寫成：
$$ f(x) = w \cdot x + b = 0 $$
其中 $w$ 是權重向量（決定線的方向），$b$ 是偏差（決定線的平移）。

### 分類規則
我們利用上述函數的符號來進行二元分類：
$$ \text{預測類別} = \text{sign}(w \cdot x + b) $$
大於 0 的預測為一類（例如 +1），小於 0 的預測為另一類（例如 -1）。

### Margin 公式
經過幾何學的推導，這條分隔線到兩側最近點的距離 (Margin) 可以表示為：
$$ \text{Margin} = \frac{2}{\|w\|} $$

### SVM 最佳化問題
為了讓 Margin 最大化，我們需要最小化 $\|w\|$。
在數學上，為了方便微分，我們將目標寫為最小化 $\frac{1}{2}\|w\|^2$。

於是 SVM 的核心問題變成了：
**Minimize**  $\frac{1}{2}\|w\|^2$  
**Subject to**  $y_i(w \cdot x_i + b) \ge 1$  (也就是所有資料點都要在正確的一邊，而且不能進入 Margin 區域)

---
### 動畫：Support Vectors 的奧秘
SVM 有一個非常美麗的特性：**真正決定這條線怎麼畫的，只有那些剛好貼在 Margin 邊緣上的點！** 
這些點就叫做 **Support Vectors**。即使你把其他的點刪掉，只要 Support Vectors 還在，找出來的線就會一模一樣。
""")

video_path = "assets/videos/support_vectors_intro.mp4"
if os.path.exists(video_path):
    st.video(video_path)
else:
    st.warning("⚠️ 尚未產生教學影片！請參閱 README 使用 Manim 於本機渲染 `support_vectors_scene.py`，並放置於 `assets/videos/support_vectors_intro.mp4`。")
