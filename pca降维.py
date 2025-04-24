import pandas as pd
from sklearn.cluster import KMeans

# 1. 读取 PCA 处理后的数据
pca_data = pd.read_csv("C:\\Users\\admin\\Desktop\\pca_sanwei_data.csv")

# 查看数据前几行
print(pca_data.head())

# 2. 保留城市名称列，并删除非数值列（如城市名称等）
cities = pca_data['城市名称']  # 保留城市名称列
pca_data_numeric = pca_data.drop(columns=['城市名称'])  # 删除城市名称列，保留数值型数据

# 3. 进行 K-Means 聚类（这里选择 4 类，可调整）
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
pca_data_numeric['Cluster'] = kmeans.fit_predict(pca_data_numeric)

# 4. 添加城市名称列到聚类结果中
pca_data_numeric['城市名称'] = cities

# 定义类别标签
cluster_labels = {
    3: "农业基础型",
    0: "均衡发展型",
    2: "创新主导型",
    1: "工业驱动型"
}

# 映射类别标签
pca_data_numeric['Cluster_Label'] = pca_data_numeric['Cluster'].map(cluster_labels)

# 5. 统计每个类别的数据量
print("每个聚类的数量：\n", pca_data_numeric['Cluster'].value_counts())

# 6. 再次统计类别数量
print("聚类后的类别分布：\n", pca_data_numeric['Cluster_Label'].value_counts())

# 7. 保存聚类结果到文件
pca_data_numeric.to_csv("pca_clustered_with_labels_and_cities2.csv", index=False)
print("✅ 聚类和分类结果已保存为 pca_clustered_with_labels_and_cities.csv")
