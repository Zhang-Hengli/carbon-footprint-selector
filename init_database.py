import os
# 使用 Hugging Face 中国镜像加速下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os

EXCEL_FILE = "ecoinvent数据库信息-给AI.xlsx"
CHROMA_PATH = "./chroma_db"

def load_data():
    print("正在读取 Excel 数据...")
    df = pd.read_excel(EXCEL_FILE)
    print(f"共加载 {len(df)} 条数据")
    return df

def prepare_documents(df):
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
    return documents, ids, metadatas

def create_vector_store(documents, ids, metadatas):
    print("正在加载向量化模型...")
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
    
    # 分批插入，每批最多 5000 条
    batch_size = 5000
    total = len(documents)
    print(f"正在存储向量（共 {total} 条，分批处理）...")
    
    for i in range(0, total, batch_size):
        end_idx = min(i + batch_size, total)
        batch_docs = documents[i:end_idx]
        batch_ids = ids[i:end_idx]
        batch_embs = embeddings[i:end_idx].tolist()
        batch_metas = metadatas[i:end_idx]
        
        collection.add(
            documents=batch_docs,
            ids=batch_ids,
            embeddings=batch_embs,
            metadatas=batch_metas
        )
        print(f"  已存储 {end_idx}/{total} 条")
    
    print(f"向量数据库已保存到 {CHROMA_PATH}")
    return collection

def main():
    df = load_data()
    documents, ids, metadatas = prepare_documents(df)
    create_vector_store(documents, ids, metadatas)
    print("初始化完成！")

if __name__ == "__main__":
    main()
