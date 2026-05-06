import pandas as pd
import chromadb

# 用 header=None 读取（不含表头，这样行号直接对应）
df = pd.read_excel('ecoinvent数据库信息-给AI.xlsx', header=None)

client = chromadb.PersistentClient(path='c:/Users/zhlco/CodeBuddy/20260423091559/chroma_db')
collection = client.get_collection('ecoinvent')

# 用户给的序号是含表头的行号，转为 DataFrame 索引 = 行号 - 1
user_numbers = [3551, 5235]
indices = [n - 1 for n in user_numbers]

print("=== 用户给的序号 vs ChromaDB 查询结果 ===\n")

for user_num, idx in zip(user_numbers, indices):
    row = df.iloc[idx]
    chroma_id = str(idx)  # ChromaDB ID = DataFrame 索引
    result = collection.get(ids=[chroma_id])
    
    print(f"用户序号(Excel行号) {user_num}:")
    print(f"  Excel内容: Activity={row[0]}, Product={row[5]}")
    print(f"  ChromaDB ID {chroma_id}: Activity={result['metadatas'][0]['activity_name']}")
    print()
