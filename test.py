import streamlit as st
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# ==========================================
# 1. 配置与缓存 (关键优化)
# ==========================================
st.set_page_config(page_title="Lung Cancer PR Predictor", layout="wide")

@st.cache_resource
def load_model():
    """缓存模型加载，防止每次刷新网页都重读文件"""
    model = CatBoostClassifier()
    # 确保文件也在 GitHub 仓库中
    model.load_model("best_model_catboost.cbm")
    return model

@st.cache_data
def load_training_data():
    """加载训练数据用于 LIME 初始化"""
    # 必须加载真实数据，否则 LIME 的解释是错误的
    # 这里的 nrows=500 是为了加速，取一部分样本即可代表分布
    try:
        df = pd.read_csv("train_data-after.csv", nrows=500)
        # 简单的列名清洗，确保和模型一致
        df.columns = df.columns.str.replace(' ', '.', regex=False).str.replace('_', '.', regex=False)
        return df
    except FileNotFoundError:
        return None

# 加载资源
model = load_model()
df_train = load_training_data()

# 获取模型真实的特征名（最稳妥的方式）
model_feature_names = model.feature_names_

# ==========================================
# 2. 侧边栏：输入参数
# ==========================================
st.sidebar.header("Patient Clinical Features")

def user_input_features():
    # 使用侧边栏让主界面更干净
    DM = st.sidebar.selectbox("Diabetes (DM)", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    Thyroid = st.sidebar.selectbox("Thyroid Dysfunction", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    Sex = st.sidebar.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x==1 else "Female")
    TNM = st.sidebar.slider("TNM Stage", 1, 4, 3)
    TTD = st.sidebar.number_input("Time to Treatment (TTD days)", 0, 365, 10)
    Multidrug = st.sidebar.selectbox("Multidrug Therapy Count", options=[0, 1, 2, 3])
    Surgery = st.sidebar.selectbox("Surgery History", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    
    # 数值型变量：建议根据临床范围设定 min/max
    Hb = st.sidebar.slider("Hemoglobin (Hb)", 50.0, 180.0, 100.0)
    TBIL = st.sidebar.slider("Total Bilirubin (TBIL)", 0.0, 100.0, 10.0)
    Cr = st.sidebar.slider("Creatinine (Cr)", 0.0, 500.0, 70.0)

    # 组装数据，键名必须与模型特征名稍微对应，后续会强制对齐
    data = {
        "DM": DM,
        "Thyroid.dysfunction": Thyroid, # 预判可能带点
        "Sex": Sex,
        "TNM": TNM,
        "TTD": TTD,
        "Multidrug.therapy": Multidrug, # 预判可能带点
        "Surgery": Surgery,
        "Hb": Hb,
        "TBIL": TBIL,
        "Cr": Cr
    }
    
    # 这里的键名其实不重要，重要的是下面的对齐步骤
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# ==========================================
# 3. 数据对齐 (防止特征名报错)
# ==========================================
# 创建一个符合模型顺序的 DataFrame
X_input = pd.DataFrame(index=[0])

# 你的手动输入可能没有覆盖所有特征名，或者名字有细微差别(空格vs点)
# 这里做一个映射处理，确保输入值能填入模型需要的列
feature_map = {
    "DM": "DM",
    "Thyroid": "Thyroid.dysfunction", # 假设模型用的是带点的
    "Sex": "Sex",
    "TNM": "TNM",
    "TTD": "TTD",
    "Multidrug": "Multidrug.therapy",
    "Surgery": "Surgery",
    "Hb": "Hb",
    "TBIL": "TBIL",
    "Cr": "Cr"
}

# 自动填充
for col in model_feature_names:
    # 尝试直接匹配
    if col in input_df.columns:
        X_input[col] = input_df[col]
    # 尝试模糊匹配 (比如输入是 Thyroid.dysfunction，模型是 Thyroid dysfunction)
    else:
        # 这里为了演示简单，如果找不到对应列，默认填0，实际需要你根据模型真实名字调整上方 data 字典
        # print(f"Warning: {col} not found in input, filling 0")
        X_input[col] = input_df.get(col, 0) # 尝试获取，没有则0

# ==========================================
# 4. 主界面：预测与结果
# ==========================================
st.title("🧬 PD-1 Lung Cancer Response Predictor")
st.markdown("Predict the probability of **Partial Response (PR)** based on clinical features.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction Result")
    if st.button("Run Prediction", type="primary"):
        # 预测
        pred_class = model.predict(X_input)[0]
        pred_proba = model.predict_proba(X_input)[0][1]
        
        # 显示大号结果
        if pred_class == 1:
            st.success(f"**Outcome: Partial Response (PR)**")
        else:
            st.warning(f"**Outcome: Non-PR**")
            
        st.metric(label="Probability of PR", value=f"{pred_proba:.2%}")

        # -------------------------------
        # SHAP 可解释性 (推荐用 Waterfall)
        # -------------------------------
        st.subheader("🔍 SHAP Explanation")
        with st.spinner("Calculating SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_input)
            
            # 使用 Waterfall Plot，它是静态的，Streamlit 支持得更好
            fig_shap, ax = plt.subplots(figsize=(8, 6))
            # [0, :, 1] 取第一个样本，所有特征，正类(1)的SHAP值
            # 注意: CatBoost 的 explainer output 结构可能因版本而异
            # 如果报错，尝试 shap_values[0]
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(fig_shap)

        # -------------------------------
        # LIME 可解释性
        # -------------------------------
        st.subheader("🍋 LIME Explanation")
        if df_train is not None:
            # 确保 LIME 使用的训练数据只包含模型需要的特征
            X_train_lime = df_train[model_feature_names].fillna(0)
            
            lime_explainer = LimeTabularExplainer(
                training_data=np.array(X_train_lime),
                feature_names=model_feature_names,
                class_names=['Non-PR', 'PR'],
                mode='classification',
                verbose=False
            )
            
            lime_exp = lime_explainer.explain_instance(
                data_row=np.array(X_input)[0],
                predict_fn=model.predict_proba
            )
            
            # 直接显示 LIME 的图形 html
            st.components.v1.html(lime_exp.as_html(), height=400, scrolling=True)
        else:
            st.error("Training data (csv) not found. Cannot run LIME.")

with col2:
    st.write("### Current Input Data")
    st.dataframe(X_input.T)
