import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib

# 设置 Matplotlib 以支持中文
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# 1️⃣ 读取降维后的数据（假设你已经完成了PCA降维）
pca_data = pd.read_csv("C:\\Users\\admin\\Desktop\\pca_clustered_data.csv")  # 替换为你的文件路径

# 2️⃣ 选择要进行聚类的特征列
# 使用降维后的前三个主成分
X = pca_data[["主成分1", "主成分2", "主成分3"]]  # 使用前三个主成分

# 3️⃣ 使用 K-means 聚类（K=4）
kmeans = KMeans(n_clusters=4, random_state=42, n_init=20, max_iter=500, tol=1e-5)
pca_data['Cluster'] = kmeans.fit_predict(X)

# 4️⃣ 将聚类结果映射为标签（根据需要命名类）
cluster_labels = {0: "均衡发展型", 2: "工业驱动型", 1: "创新主导型", 3: "农业主导型"}
pca_data['Cluster_Label'] = pca_data['Cluster'].map(cluster_labels)

# 5️⃣ 统计每个聚类的数量
print(pca_data['Cluster_Label'].value_counts())

# 6️⃣ 查看每个聚类的经济特征（即每个聚类的均值），但排除非数值列
numeric_columns = pca_data.select_dtypes(include=['number']).columns
cluster_means = pca_data[numeric_columns].groupby('Cluster').mean()
print("\n每个聚类的经济特征（均值）：")
print(cluster_means)

# 7️⃣ 可视化聚类结果（使用主成分1、主成分2和主成分3）
# 这里选择了3D散点图来展示聚类结果
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制3D散点图
ax.scatter(pca_data['主成分1'], pca_data['主成分2'], pca_data['主成分3'], c=pca_data['Cluster'], cmap='viridis', alpha=0.7)

ax.set_xlabel('主成分1')
ax.set_ylabel('主成分2')
ax.set_zlabel('主成分3')
plt.title('PCA降维后的聚类结果（3D）')
plt.show()

# 8️⃣ 保存聚类结果
pca_data.to_csv("pca_clustered_with_labels1.csv", index=True)
print("✅ 聚类完成，已保存为 pca_clustered_with_labels.csv")
