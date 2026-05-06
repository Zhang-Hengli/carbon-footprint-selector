import os
# 使用 Hugging Face 中国镜像加速下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL = 'qwen2.5:7b'

def load_vector_store():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma_client.get_collection("ecoinvent")

def search_factors(collection, query, n_results=5):
    print(f"正在检索: {query}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    
    return results

def generate_recommendation(user_input, search_results):
    docs = search_results['documents'][0]
    metadatas = search_results['metadatas'][0]
    
    context = "\n\n".join([
        f"【候选{i+1}】\n活动名称: {meta.get('activity_name', 'N/A')}\n产品名称: {meta.get('product_name', 'N/A')}\n行业: {meta.get('sector', 'N/A')}\n详细描述: {doc[:2000]}..."
        for i, (doc, meta) in enumerate(zip(docs, metadatas))
    ])
    
    prompt = f"""你是一个专业的LCA（生命周期评估）碳足迹因子选择助手。用户正在为碳足迹建模选择合适的生态库因子。

用户输入：{user_input}

以下是语义检索到的最相关的因子候选：

{context}

请根据用户输入，从上述候选中选择最匹配的1-3个因子，并说明匹配理由。

输出格式：
1. 推荐因子：[活动名称] - [匹配理由]
2. 匹配度：[高/中/低]
3. 适用场景：[说明]
"""

    print("正在生成AI推荐...")
    response = ollama.chat(model=LLM_MODEL, messages=[
        {"role": "system", "content": "你是一个专业的LCA碳足迹因子选择助手，擅长从生态库中选择最合适的过程因子。"},
        {"role": "user", "content": prompt}
    ])
    
    return response['message']['content']

def main():
    collection = load_vector_store()
    print("=" * 60)
    print("AI 碳足迹因子选择器")
    print("=" * 60)
    print("输入 'quit' 退出\n")
    
    while True:
        user_input = input("请输入原材料/产品名称（可附加工艺描述）：\n> ")
        
        if user_input.lower() == 'quit':
            print("再见！")
            break
        
        if not user_input.strip():
            continue
        
        results = search_factors(collection, user_input, n_results=5)
        recommendation = generate_recommendation(user_input, results)
        
        print("\n" + "=" * 60)
        print("AI 推荐结果")
        print("=" * 60)
        print(recommendation)
        print("\n")

if __name__ == "__main__":
    main()
