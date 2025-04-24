# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
from chardet import detect


def clean_column_names(col):
    """统一列名格式（处理中英文括号和空格）"""
    col = re.sub(r'[（）]', '', col).strip()  # 去除中文括号
    col = re.sub(r'\s+', '', col)  # 去除所有空格
    # 关键字段标准化
    if re.search(r'gdp.*?亿元.*?人民币', col, re.IGNORECASE):
        return 'gdp(亿元人民币)'
    elif re.search(r'第一产业.*?GDP比重', col):
        return '第一产业占GDP比重'
    elif re.search(r'第二产业.*?GDP比重', col):
        return '第二产业占GDP比重'
    elif re.search(r'第三产业.*?GDP比重', col):
        return '第三产业占GDP比重'
    return col


def handle_missing_values(file_path):
    """完整的缺失值处理流程"""
    # 检测文件编码
    with open(file_path, 'rb') as f:
        encoding = detect(f.read(10000))['encoding']

    # 加载数据（兼容中文编码）
    raw_df = pd.read_csv(file_path, encoding=encoding)
    raw_df.columns = [clean_column_names(col) for col in raw_df.columns]

    # 定义关键字段（函数内部作用域）
    numeric_cols = [
        'gdp(亿元人民币)',
        '第一产业增加值亿元人民币',
        '第二产业增加值',
        '第三产业增加值',
        '第一产业占GDP比重',
        '第二产业占GDP比重',
        '第三产业占GDP比重'
    ]

    # 字段存在性校验
    missing_cols = [col for col in numeric_cols if col not in raw_df.columns]
    if missing_cols:
        raise KeyError(f"缺失关键字段: {missing_cols}，请检查原始数据列名")

    # 处理前缺失统计
    print("\n=== 处理前缺失统计 ===")
    print(raw_df[numeric_cols].isnull().sum())

    # 复制数据用于处理
    df = raw_df.copy()

    # === 分阶段缺失处理 ===
    # 第一阶段：按城市时间序列插值
    df = df.sort_values(['城市名称', '年度标识'])
    for col in numeric_cols:
        df[col] = df.groupby('城市名称', group_keys=False)[col].apply(
            lambda x: x.interpolate(method='linear', limit=2)
            .fillna(x.median())  # 处理边缘缺失
        )

    # 第二阶段：省份-年度联合填充
    for col in numeric_cols:
        df[col] = df.groupby(['省份名称', '年度标识'])[col].transform(
            lambda x: x.fillna(x.mean()) if x.count() >= 3 else x.fillna(x.median())
        )

    # 第三阶段：全局兜底处理
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df = df.dropna(subset=numeric_cols)

    # 处理后统计
    print("\n=== 处理后缺失统计 ===")
    print(df[numeric_cols].isnull().sum())

    # 保存结果
    output_path = file_path.replace('.csv', '_cleaned.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    return df


if __name__ == "__main__":
    # 使用原始字符串处理Windows路径
    input_path = r"C:\Users\admin\Desktop\地级市数据1.csv"
    try:
        cleaned_df = handle_missing_values(input_path)
        print(f"\n处理完成！清洁数据已保存至：{input_path.replace('.csv', '_cleaned.csv')}")
        print("\n示例数据（前3行）：")
        print(cleaned_df.head(3))
    except Exception as e:
        print(f"处理失败：{str(e)}")