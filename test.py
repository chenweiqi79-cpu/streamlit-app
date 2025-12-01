from catboost import CatBoostClassifier

# 加载 R 训练的模型
model = CatBoostClassifier()
model.load_model("best_model_catboost.cbm")

# 输入特征顺序一定要和 R 的训练顺序一致
import numpy as np
X = np.array([[1, 0, 1, 2, 100, 1, 1, 120, 10, 80]])  # 示例特征

pred = model.predict_proba(X)
print(pred)

git init
