import pandas as pd

df = pd.read_excel('ecoinvent数据库信息-给AI.xlsx')

with open('analysis.txt', 'w', encoding='utf-8') as f:
    f.write('=== 列名 ===\n')
    f.write(str(df.columns.tolist()) + '\n\n')
    f.write('=== 数据形状 ===\n')
    f.write(f'总行数: {len(df)}\n')
    f.write(f'总列数: {len(df.columns)}\n\n')
    f.write('=== 前3行数据 ===\n')
    f.write(df.head(3).to_string())
    f.write('\n\n')
    f.write('=== 各列数据类型 ===\n')
    f.write(str(df.dtypes))
