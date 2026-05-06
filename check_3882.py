import pandas as pd
import chromadb

df = pd.read_excel('ecoinvent数据库信息-给AI.xlsx', header=None)
client = chromadb.PersistentClient(path='c:/Users/zhlco/CodeBuddy/20260423091559/chroma_db')
collection = client.get_collection('ecoinvent')

# 用户给的是Excel行号3882，需要-1转为ChromaDB ID
chroma_id = '3881'
result = collection.get(ids=[chroma_id])
print(f"ChromaDB ID 3881 (Excel行号 3882):")
print(f"  Product: {result['metadatas'][0].get('product_name', 'N/A')}")
print(f"  Activity: {result['metadatas'][0].get('activity_name', 'N/A')}")
