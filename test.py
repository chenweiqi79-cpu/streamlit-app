import streamlit as st
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# ==========================================
# 1. 配置与缓存
# ==========================================
st.set_page_config(page_title="Lung Cancer Objective Response Predictor", layout="wide")

@st.cache_resource
def load_model():
    """缓存模型加载"""
    model = CatBoostClassifier()
    # 确保 best_model_catboost.cbm 文件在 GitHub 仓库根目录
    model.load_model("best_model_catboost.cbm")
    return model

@st.cache_data
def load_training_data():
    """加载训练数据用于 LIME 初始化"""
    try:
        df = pd.read_csv("train_data-after.csv", nrows=500)
        # 清洗列名以匹配模型
        df.columns = df.columns.str.replace(' ', '.', regex=False).str.replace('_', '.', regex=False)
        return df
    except FileNotFoundError:
        return None

# 加载模型和数据
model = load_model()
df_train = load_training_data()
model_feature_names = model.feature_names_

# ==========================================
# 2. 侧边栏：输入参数 (TTD 单位改为 Months)
# ==========================================
st.sidebar.header("Patient Clinical Features")

def user_input_features():
    DM = st.sidebar.selectbox("Diabetes (DM)", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    Thyroid = st.sidebar.selectbox("Thyroid Dysfunction", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    Sex = st.sidebar.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x==1 else "Female")
    TNM = st.sidebar.slider("TNM Stage", 1, 4, 3)
    
    # 修改点：单位从 days 改为 Months
    TTD = st.sidebar.number_input("Tumor Treatment Duration (TTD months)", 0, 60, 15) 
    
    Multidrug = st.sidebar.selectbox("Multidrug Therapy Count", options=[0, 1, 2, 3])
    Surgery = st.sidebar.selectbox("Surgery History", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    Hb = st.sidebar.slider("Hemoglobin (Hb)", 50.0, 180.0, 113.0)
    TBIL = st.sidebar.slider("Total Bilirubin (TBIL)", 0.0, 100.0, 10.0)
    Cr = st.sidebar.slider("Creatinine (Cr)", 0.0, 500.0, 73.0)

    data = {
        "DM": DM,
        "Thyroid.dysfunction": Thyroid,
        "Sex": Sex,
        "TNM": TNM,
        "TTD": TTD,
        "Multidrug.therapy": Multidrug,
        "Surgery": Surgery,
        "Hb": Hb,
        "TBIL": TBIL,
        "Cr": Cr
    }
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ==========================================
# 3. 数据对齐
# ==========================================
X_input = pd.DataFrame(index=[0])
for col in model_feature_names:
    if col in input_df.columns:
        X_input[col] = input_df[col]
    else:
        X_input[col] = input_df.get(col, 0)

# ==========================================
# 4. 主界面：预测与结果 (标题改为 Objective Response)
# ==========================================
st.title("🧬 Lung Cancer Objective Response Predictor")
st.markdown("Predict the probability of **Objective Response (OR)** based on clinical features.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction Result")
    if st.button("Run Prediction", type="primary"):
        # 预测
        pred_class = model.predict(X_input)[0]
        pred_proba = model.predict_proba(X_input)[0][1]
        
        # 显示结果
        if pred_class == 1:
            st.success(f"**Outcome: Objective Response (OR)**")
        else:
            st.warning(f"**Outcome: Non-Response**")
            
        st.metric(label="Probability of Objective Response", value=f"{pred_proba:.2%}")

        # SHAP 可解释性
        st.subheader("🔍 SHAP Explanation")
        with st.spinner("Calculating SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_input)
            fig_shap, ax = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(fig_shap)

        # LIME 可解释性
        st.subheader("🍋 LIME Explanation")
        if df_train is not None:
            X_train_lime = df_train[model_feature_names].fillna(0)
            lime_explainer = LimeTabularExplainer(
                training_data=np.array(X_train_lime),
                feature_names=model_feature_names,
                class_names=['Non-Response', 'Objective Response'],
                mode='classification'
            )
            lime_exp = lime_explainer.explain_instance(
                data_row=np.array(X_input)[0],
                predict_fn=model.predict_proba
            )
            st.components.v1.html(lime_exp.as_html(), height=400, scrolling=True)
        else:
            st.error("Training data not found. Cannot run LIME.")

with col2:
    st.write("### Current Input Data")
    # 转置显示方便查看
    st.dataframe(X_input.T)
