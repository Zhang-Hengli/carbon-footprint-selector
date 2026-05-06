import chromadb

client = chromadb.PersistentClient(path='c:/Users/zhlco/CodeBuddy/20260423091559/chroma_db')
collection = client.get_collection('ecoinvent')

result = collection.get(ids=['3551', '5235'])

for i in range(len(result['ids'])):
    meta = result['metadatas'][i]
    print(f"ID: {result['ids'][i]}")
    print(f"  Product: {meta.get('product_name', 'N/A')}")
    print(f"  Activity: {meta.get('activity_name', 'N/A')}")
    print(f"  Document: {result['documents'][i][:200] if result['documents'][i] else 'N/A'}")
    print()
