import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

# 读取数据
df = pd.read_csv("C:\\Users\\admin\\Desktop\\地级市数据1_cleaned.csv")  # 请替换为你的数据文件路径

# 选取所有数值型列
numeric_cols = df.select_dtypes(include=[np.number]).columns

# 存储 Winsorize 处理前后的统计信息
winsorized_info = {}

# 处理数据
df_winsorized = df.copy()
for col in numeric_cols:
    # 计算原始统计量
    orig_std = df[col].std()
    orig_skew = df[col].skew()

    # 进行 Winsorize 处理（上下 5%）
    df_winsorized[col] = winsorize(df[col], limits=[0.05, 0.05])

    # 计算新统计量
    new_std = df_winsorized[col].std()
    new_skew = df_winsorized[col].skew()

    # 计算标准差减少百分比
    std_reduction = ((orig_std - new_std) / orig_std) * 100

    # 记录 Winsorize 处理前后的变化
    winsorized_info[col] = {
        "标准差减少百分比": round(std_reduction, 2),
        "原始偏度": round(orig_skew, 2),
        "处理后偏度": round(new_skew, 2)
    }

# 转换统计信息为 DataFrame 并打印
winsorized_info_df = pd.DataFrame(winsorized_info).T
print("\nWinsorize 处理前后的统计信息：")
print(winsorized_info_df)

# 保存处理后的数据
df_winsorized.to_csv("processed_data_winsorized.csv", index=False)
print("\n数据处理完成！已保存至 processed_data_winsorized.csv")
