import os
# 使用 Hugging Face 中国镜像加速下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import os

EXCEL_FILE = "ecoinvent数据库信息-给AI.xlsx"
CHROMA_PATH = "./chroma_db"

# 字段权重配置
FIELD_WEIGHTS = {
    "CAS Number": 0.35,        # 最高权重
    "Reference Product Name": 0.30,  # 高权重
    "Activity Name": 0.20,     # 中权重
    "Product Information": 0.10,     # 低权重
    "Process description": 0.05     # 最低权重
}

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
        # 提取各字段
        activity_name = str(row.get("Activity Name", "")) if pd.notna(row.get("Activity Name")) else ""
        product_name = str(row.get("Reference Product Name", "")) if pd.notna(row.get("Reference Product Name")) else ""
        sector = str(row.get("Sector", "")) if pd.notna(row.get("Sector")) else ""
        cas_number = str(row.get("CAS Number", "")) if pd.notna(row.get("CAS Number")) else ""
        product_info = str(row.get("Product Information", "")) if pd.notna(row.get("Product Information")) else ""
        process_desc = str(row.get("Process description", "")) if pd.notna(row.get("Process description")) else ""
        cut_off = str(row.get("Cut-Off Classification", "")) if pd.notna(row.get("Cut-Off Classification")) else ""
        special_activity = str(row.get("Special Activity Type", "")) if pd.notna(row.get("Special Activity Type")) else ""
        
        # 构建元数据
        metadata = {
            "activity_name": activity_name,
            "product_name": product_name,
            "sector": sector,
            "cas_number": cas_number,
            "cut_off": cut_off,
            "special_activity": special_activity
        }
        
        # 构建文档文本（用于向量化和 LLM 读取）
        doc_parts = []
        if cas_number:
            doc_parts.append(f"CAS号: {cas_number}")
        if product_name:
            doc_parts.append(f"产品名称: {product_name}")
        if activity_name:
            doc_parts.append(f"活动名称: {activity_name}")
        if sector:
            doc_parts.append(f"行业: {sector}")
        if cut_off:
            doc_parts.append(f"截止分类: {cut_off}")
        if special_activity:
            doc_parts.append(f"特殊活动类型: {special_activity}")
        if product_info:
            doc_parts.append(f"产品信息: {product_info}")
        if process_desc:
            doc_parts.append(f"详细描述: {process_desc}")
        
        document = " | ".join(doc_parts)
        
        documents.append(document)
        ids.append(str(idx + 1))
        metadatas.append(metadata)
    
    print(f"准备完成 {len(documents)} 条文档")
    return documents, ids, metadatas

def get_field_embeddings(documents, model):
    """获取每个字段的向量化"""
    print("正在获取 CAS 编号向量...")
    cas_docs = [d.split("CAS号:")[1].split("|")[0].strip() if "CAS号:" in d else "" for d in documents]
    
    print("正在获取产品名称向量...")
    product_docs = [d.split("产品名称:")[1].split("|")[0].strip() if "产品名称:" in d else "" for d in documents]
    
    print("正在获取活动名称向量...")
    activity_docs = [d.split("活动名称:")[1].split("|")[0].strip() if "活动名称:" in d else "" for d in documents]
    
    print("正在获取产品信息向量...")
    info_docs = [d.split("产品信息:")[1].split("|")[0].strip() if "产品信息:" in d else "" for d in documents]
    
    print("正在获取详细描述向量...")
    desc_docs = [d.split("详细描述:")[1][:200] if "详细描述:" in d else "" for d in documents]
    
    cas_emb = model.encode(cas_docs, show_progress_bar=True)
    product_emb = model.encode(product_docs, show_progress_bar=True)
    activity_emb = model.encode(activity_docs, show_progress_bar=True)
    info_emb = model.encode(info_docs, show_progress_bar=True)
    desc_emb = model.encode(desc_docs, show_progress_bar=True)
    
    return {
        "CAS Number": cas_emb,
        "Reference Product Name": product_emb,
        "Activity Name": activity_emb,
        "Product Information": info_emb,
        "Process description": desc_emb
    }

def create_vector_store(documents, ids, metadatas):
    print("正在加载向量化模型...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("正在生成加权向量...")
    field_embeddings = get_field_embeddings(documents, model)
    
    # 加权组合向量
    combined_embeddings = np.zeros_like(field_embeddings["CAS Number"])
    for field, weight in FIELD_WEIGHTS.items():
        combined_embeddings += field_embeddings[field] * weight
    
    print("正在创建 ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    try:
        chroma_client.delete_collection("ecoinvent")
    except:
        pass
    
    collection = chroma_client.create_collection("ecoinvent")
    
    print("正在存储向量...")
    
    # 分批添加，每批最多 5000 条
    batch_size = 5000
    total = len(documents)
    for i in range(0, total, batch_size):
        end_idx = min(i + batch_size, total)
        print(f"  存储进度: {end_idx}/{total}")
        collection.add(
            documents=documents[i:end_idx],
            ids=ids[i:end_idx],
            embeddings=combined_embeddings[i:end_idx].tolist(),
            metadatas=metadatas[i:end_idx]
        )
    
    print(f"向量数据库已保存到 {CHROMA_PATH}")
    return collection

def main():
    df = load_data()
    documents, ids, metadatas = prepare_documents(df)
    create_vector_store(documents, ids, metadatas)
    print("初始化完成！")

if __name__ == "__main__":
    main()
