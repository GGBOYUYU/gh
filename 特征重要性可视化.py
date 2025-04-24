import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
# 设置 Matplotlib 以支持中文
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示问题
# 1. 加载数据
# 假设你的数据文件包含PCA降维后的三个主成分（PC1, PC2, PC3）以及聚类标签（Cluster_Labels）
data = pd.read_csv("C:\\Users\\admin\\Desktop\\pca_clustered_with_labels1.csv")  # 替换为你的实际文件路径

# 打印前几行以确认数据正确加载
print(data.head())

# 2. 特征选择：这里我们选择PCA降维后的三个特征
features = ['主成分1', '主成分2', '主成分3']  # 确保这些列名与你的数据集中的列名匹配
X = data[features]

# 目标变量：'Cluster_Labels' 列是你想要预测的目标变量
y = data['Cluster_Label']

# 3. 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 初始化随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 使用训练集训练模型
rf_classifier.fit(X_train, y_train)

# 5. 预测测试集结果
y_pred = rf_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}")

# 输出分类报告
print("\n分类报告：")
print(classification_report(y_test, y_pred))

# 打印特征重要性
feature_importances = pd.DataFrame(rf_classifier.feature_importances_,
                                   index=X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print("\n特征重要性：")
print(feature_importances)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y=feature_importances.index, data=feature_importances, palette="viridis")
plt.title('特征重要性')
plt.xlabel('重要性')
plt.ylabel('特征')
plt.tight_layout()

# 显示图表
plt.show()

# 可选：保存带预测标签的数据到新的CSV文件
data['Predicted_Cluster_Labels'] = rf_classifier.predict(X)
data.to_csv("processed_city_data_with_predictions.csv", index=False)