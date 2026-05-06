import pandas as pd

df = pd.read_excel('ecoinvent数据库信息-给AI.xlsx')

# 查看Excel行号3551的内容（DataFrame索引 = Excel行号 - 1）
excel_row_3551 = df.iloc[3550]

print(f"Excel 行号 3551 (DataFrame 索引 3550):")
print(f"  Activity Name: {excel_row_3551['Activity Name']}")
print(f"  Reference Product: {excel_row_3551['Reference Product Name']}")

# 查看索引3550-3552附近的行
print("\n--- 索引 3548-3552 附近内容 ---")
for idx in range(3548, 3553):
    row = df.iloc[idx]
    print(f"DataFrame索引 {idx} (Excel行号 {idx+1}): {row['Reference Product Name'][:50]}...")
