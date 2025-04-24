import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
# 设置 Matplotlib 以支持中文
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示问题
# 加载你的数据
file_path = 'C:\\Users\\admin\\Desktop\\pca_clustered_with_labels.csv'
pca_data = pd.read_csv(file_path)

# 将数据从宽格式转换为长格式，以便于绘图
pca_long = pd.melt(pca_data, id_vars=['Cluster_Label'], value_vars=['主成分1', '主成分2', '主成分3'],
                    var_name='Principal_Component', value_name='Score')

# 绘制箱线图
plt.figure(figsize=(15, 8))
sns.boxplot(x='Principal_Component', y='Score', hue='Cluster_Label', data=pca_long)

# 添加标题和标签
plt.title('不同类别在各主成分上的分布')
plt.xlabel('主成分')
plt.ylabel('得分')
plt.legend(title='聚类类别')

# 显示图形
plt.show()