import chromadb
import pandas as pd

client = chromadb.PersistentClient(path='c:/Users/zhlco/CodeBuddy/20260423091559/chroma_db')
collection = client.get_collection('ecoinvent')

# 查询 ChromaDB ID 3551
result = collection.get(ids=['3551'])
print(f"ChromaDB ID 3551:")
print(f"  Activity: {result['metadatas'][0].get('activity_name', 'N/A')}")
print(f"  Product: {result['metadatas'][0].get('product_name', 'N/A')}")

# 查询 Excel 行号 3551
df = pd.read_excel('ecoinvent数据库信息-给AI.xlsx')
excel_row = df.iloc[3550]  # Excel 行号 3551 = DataFrame 索引 3550
print(f"\nExcel 行号 3551:")
print(f"  Activity: {excel_row['Activity Name']}")
print(f"  Product: {excel_row['Reference Product Name']}")

# 对比
chroma_activity = result['metadatas'][0].get('activity_name', 'N/A')
excel_activity = excel_row['Activity Name']
print(f"\n是否一致: {chroma_activity == excel_activity}")
