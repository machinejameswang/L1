import streamlit as st

st.set_page_config(page_title="SVM 測驗", page_icon="📝")

st.title("第五章：SVM 概念小測驗")
st.markdown("測試看看你對前面幾個章節的吸收程度吧！")

questions = [
    {
        "question": "1. Support Vectors (支持向量) 是什麼？",
        "options": [
            "所有訓練資料的平均值",
            "距離決策邊界最遠的資料點",
            "距離決策邊界最近的資料點，真正決定邊界位置的點",
            "被分類錯誤的所有資料點"
        ],
        "answer": "距離決策邊界最近的資料點，真正決定邊界位置的點",
        "explanation": "正確！Support Vectors 是恰好落在 Margin 邊緣上的點。如果你拿掉這些點，決策邊界就會改變；但拿掉其他點則不影響。"
    },
    {
        "question": "2. 參數 C (Regularization) 變大時，通常會造成什麼效果？",
        "options": [
            "容忍更多錯誤，Margin 變寬，可能 Underfitting",
            "不容忍錯誤，Margin 變窄，可能 Overfitting",
            "資料維度會增加",
            "訓練速度變快"
        ],
        "answer": "不容忍錯誤，Margin 變窄，可能 Overfitting",
        "explanation": "正確！C 是懲罰係數。C 越大，模型越不允許點落入錯誤區域，導致邊界變得扭曲且 Margin 變窄，容易導致 Overfitting。"
    },
    {
        "question": "3. 在 RBF Kernel 中，Gamma 變大通常會造成什麼效果？",
        "options": [
            "每個點的影響範圍變大，邊界變平滑",
            "每個點的影響範圍變小，邊界變得非常曲折、容易過度包覆單一點",
            "會把資料轉回 2D",
            "對模型完全沒有影響"
        ],
        "answer": "每個點的影響範圍變小，邊界變得非常曲折、容易過度包覆單一點",
        "explanation": "正確！Gamma 決定了 RBF 的高斯分佈寬度。Gamma 越大，分布越窄（影響範圍小），模型只能「局部」擬合，導致 Overfitting。"
    },
    {
        "question": "4. SVM 的目標是最大化 Margin。Margin 越大代表什麼？",
        "options": [
            "模型對未知的測試資料更有信心，泛化能力 (Generalization) 較好",
            "模型的訓練錯誤率一定會變成 0",
            "運算速度會變慢",
            "表示資料一定不是線性可分的"
        ],
        "answer": "模型對未知的測試資料更有信心，泛化能力 (Generalization) 較好",
        "explanation": "正確！Margin 越寬，表示兩類資料之間的緩衝區越大，未來如果出現一些雜訊資料，也比較不容易被錯誤分類。"
    },
    {
        "question": "5. RBF Kernel 適合處理什麼樣的資料？",
        "options": [
            "只有線性可分的資料",
            "非線性可分、形狀複雜的資料 (如：同心圓)",
            "只有文字資料",
            "只有一維資料"
        ],
        "answer": "非線性可分、形狀複雜的資料 (如：同心圓)",
        "explanation": "正確！RBF Kernel 利用距離相似度，可以在高維空間中輕鬆切割出原本在 2D 複雜交疊的資料群。"
    }
]

score = 0
submitted = False

with st.form("quiz_form"):
    user_answers = []
    for idx, q in enumerate(questions):
        st.subheader(q["question"])
        # We add an empty option so nothing is selected by default
        ans = st.radio(f"選擇答案 {idx+1}", ["(請選擇)"] + q["options"], key=f"q_{idx}")
        user_answers.append(ans)
        st.write("---")
        
    submitted = st.form_submit_button("送出答案")

if submitted:
    st.markdown("## 測驗結果")
    for idx, (q, ans) in enumerate(zip(questions, user_answers)):
        if ans == q["answer"]:
            st.success(f"第 {idx+1} 題：答對了！🎉")
            st.info(q["explanation"])
            score += 1
        elif ans == "(請選擇)":
            st.warning(f"第 {idx+1} 題：你沒有作答。")
            st.info(f"正確答案是：**{q['answer']}**\n\n{q['explanation']}")
        else:
            st.error(f"第 {idx+1} 題：答錯了！你的答案是 '{ans}'。")
            st.info(f"正確答案是：**{q['answer']}**\n\n{q['explanation']}")
            
    st.subheader(f"總得分：{score} / {len(questions)}")
    
    if score == len(questions):
        st.balloons()
        st.success("太棒了！你已經完全掌握 SVM 的核心概念！")
        st.markdown("### 下一步建議")
        st.markdown("你可以回到 **Interactive SVM** 頁面，嘗試匯入你自己的真實資料，或是挑戰更複雜的參數組合！")
    elif score >= 3:
        st.info("做的不錯！有一些小細節可以再回到前面的章節複習一下。")
    else:
        st.warning("看來 SVM 對你來說還有點陌生，建議再回去看一次教學動畫喔！")
