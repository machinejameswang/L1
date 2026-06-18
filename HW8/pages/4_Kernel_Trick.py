import streamlit as st
import os

st.set_page_config(page_title="Kernel Trick", page_icon="🪄", layout="wide")

st.title("第四章：Kernel Trick 魔法")

st.markdown(r"""
### 為什麼需要 Kernel Trick？
在前面的線性分類中，我們只能畫出「直線」。但是，現實中的資料往往不是線性可分的！
想像一組資料長得像「同心圓」，或是互相交錯的「月牙形」。無論你怎麼畫一條直線，都無法把這兩群分乾淨。

### 投影到高維度
**Kernel Trick (核函數技巧)** 的核心概念是：
**如果我們在原本的低維度 (2D) 找不到直線可以切開資料，那我們就把資料「彈射」到高維度 (3D) 空間去！**

透過特定的數學轉換函數 $\phi(x)$，原本在 2D 看起來交纏的資料，到了 3D 可能就會高低錯落。這時，我們只要在 3D 空間中拿出一張紙（平面）一「切」，就能輕鬆將資料分開。

最神奇的是，因為有了 Kernel Trick，我們甚至不需要真的算出資料在無限高維度的座標，只要計算資料點之間的「相似度」，就能達到切分的效果，這大大節省了計算量！

---
### 動畫：2D 同心圓升級到 3D
看看 RBF Kernel 如何用 $z = \exp(-\gamma(x^2+y^2))$ 將內部的點往上拉，讓原本無法用直線分開的同心圓，在 3D 空間中可以被一個平面完美切開。
""")

video_path = "assets/videos/kernel_trick_intro.mp4"
if os.path.exists(video_path):
    st.video(video_path)
else:
    st.warning("⚠️ 尚未產生教學影片！請參閱 README 使用 Manim 於本機渲染 `kernel_trick_scene.py`，並放置於 `assets/videos/kernel_trick_intro.mp4`。")

st.markdown("""
---
### 比較不同的 Kernel
你可以在前一頁的「Interactive SVM」實驗室中嘗試不同 Kernel 的威力：
1. **Linear Kernel**：適合簡單的線性資料。無法處理同心圓或月牙。
2. **RBF (Gaussian) Kernel**：最受歡迎的萬用工具，能夠在周圍產生「光暈」般的影響，完美包覆複雜形狀。
3. **Poly (Polynomial) Kernel**：將特徵擴展到多項式空間，`degree` 決定了多項式的最高次方，越高次方能彎折越多次。
""")
