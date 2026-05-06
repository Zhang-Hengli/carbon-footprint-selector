import pandas as pd

# 测试不同的 header 设置
print("=== 测试不同的读取方式 ===")

# 方式1：不把第一行当表头（header=None）
df1 = pd.read_excel('ecoinvent数据库信息-给AI.xlsx', header=None)
print(f"\nheader=None:")
print(f"  总行数: {len(df1)}")
print(f"  行0内容: {df1.iloc[0, 0]}")  # 实际第1行
print(f"  行3550内容(0索引): {df1.iloc[3550, 0]}")  # 实际第3551行
print(f"  行3550完整: Activity={df1.iloc[3550, 0]}, Product={df1.iloc[3550, 5]}")

# 方式2：把第一行当表头（默认）
df2 = pd.read_excel('ecoinvent数据库信息-给AI.xlsx')
print(f"\n默认(header=0):")
print(f"  总行数: {len(df2)}")
print(f"  行0(表头): {df2.columns.tolist()[:3]}")  
print(f"  行0数据: {df2.iloc[0, 0]}")  # Excel第2行
print(f"  行3549数据(索引3549): Activity={df2.iloc[3549, 0]}, Product={df2.iloc[3549, 5]}")
print(f"  行3550数据(索引3550): Activity={df2.iloc[3550, 0]}, Product={df2.iloc[3550, 5]}")
