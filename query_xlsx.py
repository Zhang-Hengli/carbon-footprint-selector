import pandas as pd

df = pd.read_excel('ecoinvent数据库信息-给AI.xlsx')

print("列名:", df.columns.tolist())
print(f"\n总行数: {len(df)}")

# Excel 行号 = DataFrame 索引 + 2（因为有表头行，且索引从0开始）
indices = [3551 - 2, 5235 - 2]

for idx in indices:
    if idx < len(df):
        row = df.iloc[idx]
        print(f"\nExcel 行号 {idx + 2}:")
        for col in df.columns:
            val = row.get(col, 'N/A')
            if pd.notna(val):
                print(f"  {col}: {val}")
    else:
        print(f"Excel 行号 {idx + 2}: 超出范围（总共 {len(df)} 行）")
