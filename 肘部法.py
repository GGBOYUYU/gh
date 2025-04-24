import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib

# 设置 Matplotlib 以支持中文
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 读取降维后的数据（如果已经降维过的话）
pca_data = pd.read_csv("C:\\Users\\admin\\Desktop\\pca_sanwei_data.csv")  # 替换为你文件的路径

# 2. 选择需要进行聚类的特征列
# 假设使用主成分1到主成分3（如果你有更多维度的数据，可以按需调整）
X = pca_data[["主成分1", "主成分2", "主成分3"]]

# 3. 标准化数据（如果数据还没有标准化，KMeans 对数据尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 计算不同 K 值下的 SSE（总平方误差）
sse = []  # 用于保存每个 K 值的 SSE
K_range = range(1, 11)  # 尝试 K=1 到 K=10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)  # 获取每个 K 值的总平方误差（inertia）

# 5. 绘制肘部图
plt.figure(figsize=(8, 6))
plt.plot(K_range, sse, marker="o", linestyle='-', color='b')
plt.xlabel("K 值（聚类数量）")
plt.ylabel("总平方误差（SSE）")
plt.title("肘部图：K 值与 SSE 的关系")
plt.grid(True)
plt.show()
