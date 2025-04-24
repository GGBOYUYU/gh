import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.metrics import auc
# 设置 Matplotlib 以支持中文
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示问题

# 加载数据
data = pd.read_csv("C:\\Users\\admin\\Desktop\\pca_clustered_with_labels1.csv")  # 替换为你的实际文件路径

# 特征选择及目标变量（聚类标签）等操作保持不变
features = ['主成分1', '主成分2', '主成分3']  # 根据实际情况选择特征
X = data[features]
y = data['Cluster_Label']

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# 初始化随机森林分类器
rf_classifier = RandomForestClassifier(random_state=42)

# 创建GridSearchCV对象
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 输出最优参数
print("最优参数：", grid_search.best_params_)

# 使用最佳参数重新训练模型
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# 预测
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)  # 获取预测概率

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred))

# 计算AUC值
auc_scores = roc_auc_score(y_test, y_prob, multi_class='ovr')
print(f"\nAUC值: {auc_scores:.4f}")

# 混淆矩阵热图
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(y)), yticklabels=sorted(set(y)))
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()

# 绘制ROC曲线
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(best_rf.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test == best_rf.classes_[i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'yellow']
for i, color in zip(range(len(best_rf.classes_)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(best_rf.classes_[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()

# 应用模型对新数据进行预测 - 确保提供具体的数值
value1, value2, value3 = 0.5, 0.6, 0.7  # 替换成实际的新数据点的实际值
new_data = pd.DataFrame([[value1, value2, value3]], columns=features)
predicted_cluster = best_rf.predict(new_data)
print(f"\n预测的聚类标签: {predicted_cluster[0]}")