import streamlit as st
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# -------------------------------
# 1. 加载 CatBoost 模型
# -------------------------------
model = CatBoostClassifier()
model.load_model("best_model_catboost.cbm")

# -------------------------------
# 2. 定义特征名称
# -------------------------------
feature_names = [
    "DM", "Thyroid dysfunction", "Sex", "TNM", "TTD",
    "Multidrug therapy", "Surgery", "Hb", "TBIL", "Cr"
]

# -------------------------------
# 3. Streamlit 界面
# -------------------------------
st.title("PD-1 Lung Cancer Partial Response Predictor")

# 数值和分类输入
DM = st.selectbox("DM (0=No, 1=Yes):", options=[0, 1])
Thyroid = st.selectbox("Thyroid dysfunction (0=No, 1=Yes):", options=[0, 1])
Sex = st.selectbox("Sex (1=Male, 2=Female):", options=[1, 2])
TNM = st.number_input("TNM stage:", min_value=1, max_value=4, value=3)
TTD = st.number_input("Time to treatment (days):", min_value=0, max_value=100, value=10)
Multidrug = st.selectbox("Multidrug therapy (0-3):", options=[0, 1, 2, 3])
Surgery = st.selectbox("Surgery (0=No, 1=Yes):", options=[0, 1])
Hb = st.number_input("Hemoglobin (Hb):", min_value=0, max_value=200, value=100)
TBIL = st.number_input("Total bilirubin (TBIL):", min_value=0.0, max_value=200.0, value=10.0)
Cr = st.number_input("Creatinine (Cr):", min_value=0.0, max_value=200.0, value=70.0)

# -------------------------------
# 4. 构建特征数组
# -------------------------------
feature_values = [DM, Thyroid, Sex, TNM, TTD, Multidrug, Surgery, Hb, TBIL, Cr]
features = np.array([feature_values])

# -------------------------------
# 5. 点击预测按钮
# -------------------------------
if st.button("Predict"):
    pool = Pool(data=features, feature_names=feature_names)

    # 预测类别和概率
    predicted_class = model.predict(pool)[0]
    predicted_proba = model.predict_proba(pool)[0][1]  # 1 = PR

    st.write(f"**Predicted Class:** {'PR' if predicted_class == 1 else 'Non-PR'}")
    st.write(f"**Predicted Probability of Partial Response (PR):** {predicted_proba * 100:.2f}%")

    # -------------------------------
    # 6. SHAP 可解释性
    # -------------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    fig = plt.figure(figsize=(10, 3))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names),
                    matplotlib=True, show=False)
    st.pyplot(fig)

    # -------------------------------
    # 7. LIME 可解释性
    # -------------------------------
    # 需要训练集分布来初始化 LimeTabularExplainer
    # 如果没有训练集，可以用一些假数据近似
    dummy_train = np.random.rand(100, len(feature_names))  # 用真实训练集更好
    lime_explainer = LimeTabularExplainer(
        dummy_train,
        feature_names=feature_names,
        class_names=['Non-PR', 'PR'],
        mode='classification'
    )

    lime_exp = lime_explainer.explain_instance(
        feature_values,
        model.predict_proba,
        num_features=len(feature_names)
    )

    st.write("### LIME Explanation")
    st.dataframe(pd.DataFrame(lime_exp.as_list(), columns=['Feature', 'Contribution']))
