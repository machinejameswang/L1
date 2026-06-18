# SVM Interactive Teaching Website

這是一個專為教學 Support Vector Machine (SVM) 核心概念而設計的互動式多頁面 Streamlit 網站。內容涵蓋了數學直覺、Margin、Support Vectors 以及 Kernel Trick，並整合了預先渲染的 Manim 動畫。

## 專案結構與特性

- **Streamlit 互動介面**：即時調整 `C`、`gamma` 等超參數，透過 Plotly 畫出真實的 decision boundary。
- **Manim 動畫**：位於 `manim_scenes/` 中，僅供本機渲染，用來產生高質量的數學概念影片，並將輸出的 mp4 放置於 `assets/videos/` 供 Streamlit 播放。
- **無縫部署**：`requirements.txt` 移除了 Manim 依賴，確保能順利部署至 Streamlit Community Cloud。

## 本機執行方式

請確保您安裝了 Python (建議 3.10+)。

```powershell
# 安裝相依套件
python -m pip install -r requirements.txt

# 啟動 Streamlit 伺服器
streamlit run app.py
```

## Manim 動畫渲染方式

若您希望重新產生教學影片，請先在您的環境中安裝 Manim，並執行以下指令 (注意：請確定系統已安裝 FFmpeg 與 LaTeX 等底層依賴)：

```powershell
manim -pqh manim_scenes/svm_margin_scene.py SVMMarginScene
manim -pqh manim_scenes/support_vectors_scene.py SupportVectorsScene
manim -pqh manim_scenes/kernel_trick_scene.py KernelTrickScene
```

渲染完成後，請將輸出的 `mp4` 影片複製並重新命名到 `assets/videos/` 目錄：
- `assets/videos/svm_margin_intro.mp4`
- `assets/videos/support_vectors_intro.mp4`
- `assets/videos/kernel_trick_intro.mp4`

## 部署到 Streamlit Community Cloud

1. 將本專案推送 (Push) 到您的 GitHub 儲存庫。
2. 前往 [Streamlit Community Cloud](https://streamlit.io/) 並登入。
3. 點選 "New app"，選擇您的 GitHub repository。
4. 設定 Main file path 為 `app.py`。
5. 點擊 Deploy 即可完成部署！

## 常見問題排除

- **影片無法播放**：請確認您已在本地使用 Manim 渲染影片，並放置於 `assets/videos/` 中。如果沒有，網站會顯示提示訊息但不致崩潰。
- **Manim 安裝失敗**：這通常與作業系統底層的 C++ Build Tools 或 FFmpeg 有關。本專案將 Manim 從 `requirements.txt` 移除，因此**不會影響** Streamlit 網站本身的執行與部署。
