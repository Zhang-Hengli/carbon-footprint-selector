import pandas as pd

df = pd.read_excel('ecoinvent数据库信息-给AI.xlsx')

print(f"DataFrame 索引范围: {df.index.min()} - {df.index.max()}")
print(f"DataFrame 长度: {len(df)}")
print(f"索引是否连续: {df.index.is_monotonic_increasing}")
print(f"\n前10个索引值: {df.index[:10].tolist()}")
print(f"后10个索引值: {df.index[-10:].tolist()}")

# 检查是否有断点
missing_idx = []
expected = 0
for idx in df.index:
    if idx != expected:
        missing_idx.append((expected, idx))
    expected = idx + 1

if missing_idx:
    print(f"\n发现索引断点（前5个）:")
    for i, (exp, got) in enumerate(missing_idx[:5]):
        print(f"  期望 {exp}, 实际 {got}, 缺失 {got - exp} 个")
else:
    print("\n索引完全连续")
