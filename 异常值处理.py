import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv("C:\\Users\\admin\\Desktop\\地级市数据1_cleaned.csv")

# 选择数值列（排除文本列）
numeric_cols = df.select_dtypes(include=[np.number]).columns

# 存储异常值数量
iqr_outlier_counts = {}

# 计算每一列的异常值数量
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)  # 第一四分位数
    Q3 = df[col].quantile(0.75)  # 第三四分位数
    IQR = Q3 - Q1  # 四分位距

    # 计算异常值（低于 Q1 - 1.5*IQR 或 高于 Q3 + 1.5*IQR）
    iqr_outlier_counts[col] = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()

# 转换为 DataFrame，并按异常值数量降序排序
iqr_outlier_df = pd.DataFrame.from_dict(iqr_outlier_counts, orient='index', columns=['异常值数量'])
iqr_outlier_df = iqr_outlier_df.sort_values(by='异常值数量', ascending=False)

# 打印异常值数量
print("各列异常值数量（IQR 法）：")
print(iqr_outlier_df)

# 保存数据
iqr_outlier_df.to_csv("outlier_counts_iqr.csv", encoding="utf-8")
print("异常值数量已保存至 outlier_counts_iqr.csv")
