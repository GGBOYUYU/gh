from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # 添加此行以导入confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 加载数据
data = pd.read_csv("C:\\Users\\admin\\Desktop\\pca_clustered_with_labels.csv")  # 替换为你的实际文件路径

# 特征选择
features = ['主成分1', '主成分2', '主成分3']  # 根据实际情况选择特征
X = data[features]

# 目标变量（聚类标签）
y = data['Cluster_Label']

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf_classifier.fit(X_train, y_train)

# 预测
y_pred = rf_classifier.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred))

# 设置支持中文的字体（根据你的系统上安装的字体进行调整）
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=12)  # 使用微软雅黑作为示例

# 混淆矩阵热图
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)  # 确保confusion_matrix已正确导入
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(y)), yticklabels=sorted(set(y)))
plt.xlabel('预测标签', fontproperties=font)
plt.ylabel('真实标签', fontproperties=font)
plt.title('混淆矩阵', fontproperties=font)
plt.show()

