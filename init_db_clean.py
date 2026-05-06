import os
import shutil

# 清理所有缓存
cache_dirs = [
    "~/.cache/huggingface",
    "~/.cache/torch",
    "~/.cache/sentence_transformers"
]
for d in cache_dirs:
    path = os.path.expanduser(d)
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"已清理: {path}")

# 使用镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

EXCEL_FILE = "ecoinvent数据库信息-给AI.xlsx"
CHROMA_PATH = "./chroma_db"

def main():
    print("正在读取 Excel 数据...")
    df = pd.read_excel(EXCEL_FILE)
    print(f"共加载 {len(df)} 条数据")
    
    print("正在准备文档...")
    documents = []
    ids = []
    metadatas = []
    
    for idx, row in df.iterrows():
        text_parts = []
        metadata = {}
        
        if pd.notna(row.get("Activity Name")):
            text_parts.append(f"活动名称: {row['Activity Name']}")
            metadata["activity_name"] = str(row["Activity Name"])
        if pd.notna(row.get("Reference Product Name")):
            text_parts.append(f"产品名称: {row['Reference Product Name']}")
            metadata["product_name"] = str(row["Reference Product Name"])
        if pd.notna(row.get("Sector")):
            text_parts.append(f"行业部门: {row['Sector']}")
            metadata["sector"] = str(row["Sector"])
        if pd.notna(row.get("Process description")):
            text_parts.append(f"过程描述: {row['Process description']}")
        
        doc = " | ".join(text_parts)
        documents.append(doc)
        ids.append(str(idx + 1))
        metadatas.append(metadata)
    
    print(f"准备完成 {len(documents)} 条文档")
    
    print("正在加载向量化模型 (使用镜像)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("正在生成向量...")
    embeddings = model.encode(documents, show_progress_bar=True)
    
    print("正在创建 ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    try:
        chroma_client.delete_collection("ecoinvent")
    except:
        pass
    
    collection = chroma_client.create_collection("ecoinvent")
    
    print("正在存储向量...")
    collection.add(
        documents=documents,
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )
    
    print(f"向量数据库已保存到 {CHROMA_PATH}")
    print("初始化完成！")

if __name__ == "__main__":
    main()
