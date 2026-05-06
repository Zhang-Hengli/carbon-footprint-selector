import chromadb
import pandas as pd

client = chromadb.PersistentClient(path='c:/Users/zhlco/CodeBuddy/20260423091559/chroma_db')
collection = client.get_collection('ecoinvent')
df = pd.read_excel('ecoinvent数据库信息-给AI.xlsx')

ids = ['3551', '5235']

result = collection.get(ids=ids)

for i, id_val in enumerate(result['ids']):
    chroma_activity = result['metadatas'][i].get('activity_name', 'N/A')
    chroma_product = result['metadatas'][i].get('product_name', 'N/A')
    
    excel_idx = int(id_val) - 1
    excel_activity = df.iloc[excel_idx]['Activity Name']
    excel_product = df.iloc[excel_idx]['Reference Product Name']
    
    match = "OK" if chroma_activity == excel_activity else "FAIL"
    
    print(f"\n[{match}] ChromaDB ID {id_val}:")
    print(f"   Activity: {chroma_activity}")
    print(f"   Product: {chroma_product}")
    print(f"\n[Excel] Row {id_val}:")
    print(f"   Activity: {excel_activity}")
    print(f"   Product: {excel_product}")
