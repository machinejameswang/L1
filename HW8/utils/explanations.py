def explain_c_parameter():
    return """
    **C (Regularization parameter)** 決定了模型對「分類錯誤」的容忍度。
    - **C 很大時**：模型會盡可能把所有點分對（容忍度低），導致 margin 變窄，容易發生 **Overfitting**（過度擬合）。
    - **C 很小時**：模型允許一些點被分錯（容忍度高），藉此換取更寬的 margin，這通常會讓模型泛化能力更好，但太小可能會 **Underfitting**。
    """

def explain_gamma_parameter():
    return """
    **Gamma** 決定了單一資料點的影響範圍（在使用 RBF 或 Poly kernel 時）。
    - **Gamma 很大時**：每個點的影響範圍很小，decision boundary 會非常扭曲，只圍繞著資料點，容易 **Overfitting**。
    - **Gamma 很小時**：每個點的影響範圍很大，decision boundary 會比較平滑。
    """

def explain_kernel():
    return """
    **Kernel (核函數)** 是 SVM 的魔法，它能把資料轉換到高維度來尋找分隔線。
    - **Linear (線性)**：適合本來就可以用一條直線切開的資料。計算快。
    - **RBF (高斯核)**：最常用的非線性 kernel，能夠處理複雜彎曲的邊界，像是同心圓。
    - **Poly (多項式)**：使用多項式特徵空間，`degree` 越高，邊界可以折疊得越複雜。
    """

def explain_support_vectors():
    return """
    **Support Vectors (支持向量)** 是圖中被黃色圈起來的點。
    它們是距離 decision boundary 最近的點。**SVM 的奇妙之處在於，整條邊界完全只由這些 support vectors 決定！**
    如果你移動或刪除其他不是 support vector 的點，邊界完全不會改變。
    """
