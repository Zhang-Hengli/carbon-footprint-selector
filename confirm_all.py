import pandas as pd
import chromadb

df = pd.read_excel('ecoinvent数据库信息-给AI.xlsx', header=None)
client = chromadb.PersistentClient(path='c:/Users/zhlco/CodeBuddy/20260423091559/chroma_db')
collection = client.get_collection('ecoinvent')

ranges = [(3545, 3551), (1527, 1530), (3897, 3900), (5231, 5231), (5235, 5235), (7913, 7914)]

for start, end in ranges:
    print(f"\n=== {start}-{end} ===")
    for num in range(start, end + 1):
        idx = num - 1  # DataFrame索引 = Excel行号 - 1
        chroma_id = str(idx)
        row = df.iloc[idx]
        result = collection.get(ids=[chroma_id])
        print(f"\n[{num}] ChromaDB ID {chroma_id}:")
        print(f"  Product: {result['metadatas'][0].get('product_name', 'N/A')}")
        print(f"  Activity: {result['metadatas'][0].get('activity_name', 'N/A')}")
