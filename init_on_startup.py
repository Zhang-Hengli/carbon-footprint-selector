"""
启动脚本：检查并初始化向量数据库
"""
import os
import sys

# 使用 Hugging Face 中国镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

CHROMA_PATH = "./chroma_db"
EXCEL_FILE = "ecoinvent数据库信息-给AI.xlsx"

def check_and_init_db():
    """检查数据库是否存在，如不存在则初始化"""
    
    # 检查数据库目录是否存在且有内容
    chroma_path_exists = os.path.exists(CHROMA_PATH)
    chroma_db_file_exists = os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3"))
    
    if chroma_path_exists and chroma_db_file_exists:
        print("✅ 向量数据库已存在，跳过初始化")
        return True
    
    # 检查Excel文件是否存在
    if not os.path.exists(EXCEL_FILE):
        print(f"❌ 数据文件 {EXCEL_FILE} 不存在，无法初始化数据库")
        return False
    
    print("📦 首次部署，正在初始化向量数据库...")
    print("⏳ 这可能需要几分钟时间，请耐心等待...")
    
    # 动态导入以避免过早加载
    import pandas as pd
    import chromadb
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    # 字段权重配置
    FIELD_WEIGHTS = {
        "CAS Number": 0.35,
        "Reference Product Name": 0.30,
        "Activity Name": 0.20,
        "Product Information": 0.10,
        "Process description": 0.05
    }
    
    # 读取数据
    print("正在读取 Excel 数据...")
    df = pd.read_excel(EXCEL_FILE)
    print(f"共加载 {len(df)} 条数据")
    
    # 准备文档
    print("正在准备文档...")
    documents = []
    ids = []
    metadatas = []
    
    for idx, row in df.iterrows():
        activity_name = str(row.get("Activity Name", "")) if pd.notna(row.get("Activity Name")) else ""
        product_name = str(row.get("Reference Product Name", "")) if pd.notna(row.get("Reference Product Name")) else ""
        sector = str(row.get("Sector", "")) if pd.notna(row.get("Sector")) else ""
        cas_number = str(row.get("CAS Number", "")) if pd.notna(row.get("CAS Number")) else ""
        cut_off = str(row.get("Cut-Off Classification", "")) if pd.notna(row.get("Cut-Off Classification")) else ""
        special_activity = str(row.get("Special Activity Type", "")) if pd.notna(row.get("Special Activity Type")) else ""
        product_info = str(row.get("Product Information", "")) if pd.notna(row.get("Product Information")) else ""
        process_desc = str(row.get("Process description", "")) if pd.notna(row.get("Process description")) else ""
        
        metadata = {
            "activity_name": activity_name,
            "product_name": product_name,
            "sector": sector,
            "cas_number": cas_number,
            "cut_off": cut_off,
            "special_activity": special_activity
        }
        
        doc_parts = []
        if cas_number: doc_parts.append(f"CAS号: {cas_number}")
        if product_name: doc_parts.append(f"产品名称: {product_name}")
        if activity_name: doc_parts.append(f"活动名称: {activity_name}")
        if sector: doc_parts.append(f"行业: {sector}")
        if cut_off: doc_parts.append(f"截止分类: {cut_off}")
        if special_activity: doc_parts.append(f"特殊活动类型: {special_activity}")
        if product_info: doc_parts.append(f"产品信息: {product_info}")
        if process_desc: doc_parts.append(f"详细描述: {process_desc}")
        
        documents.append(" | ".join(doc_parts))
        ids.append(str(idx + 1))
        metadatas.append(metadata)
    
    print(f"准备完成 {len(documents)} 条文档")
    
    # 加载模型（使用镜像源）
    print("正在加载向量化模型...")
    try:
        # 尝试使用ModelScope镜像
        os.environ['HF_ENDPOINT'] = 'https://modelscope.cn/models'
        model = SentenceTransformer('Xenova/all-MiniLM-L6-v2', cache_folder='/tmp/model_cache')
    except Exception as e:
        print(f"ModelScope加载失败，尝试其他镜像: {e}")
        try:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            model = SentenceTransformer('Xenova/all-MiniLM-L6-v2', cache_folder='/tmp/model_cache')
        except Exception as e2:
            print(f"备用镜像也失败: {e2}")
            # 最后尝试直接下载
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # 生成向量
    print("正在生成加权向量...")
    print("  - CAS编号向量...")
    cas_docs = [d.split("CAS号:")[1].split("|")[0].strip() if "CAS号:" in d else "" for d in documents]
    cas_emb = model.encode(cas_docs, show_progress_bar=True)
    
    print("  - 产品名称向量...")
    product_docs = [d.split("产品名称:")[1].split("|")[0].strip() if "产品名称:" in d else "" for d in documents]
    product_emb = model.encode(product_docs, show_progress_bar=True)
    
    print("  - 活动名称向量...")
    activity_docs = [d.split("活动名称:")[1].split("|")[0].strip() if "活动名称:" in d else "" for d in documents]
    activity_emb = model.encode(activity_docs, show_progress_bar=True)
    
    print("  - 产品信息向量...")
    info_docs = [d.split("产品信息:")[1].split("|")[0].strip() if "产品信息:" in d else "" for d in documents]
    info_emb = model.encode(info_docs, show_progress_bar=True)
    
    print("  - 详细描述向量...")
    desc_docs = [d.split("详细描述:")[1][:200] if "详细描述:" in d else "" for d in documents]
    desc_emb = model.encode(desc_docs, show_progress_bar=True)
    
    # 加权组合
    combined_embeddings = cas_emb * 0.35 + product_emb * 0.30 + activity_emb * 0.20 + info_emb * 0.10 + desc_emb * 0.05
    
    # 创建数据库
    print("正在创建向量数据库...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        chroma_client.delete_collection("ecoinvent")
    except:
        pass
    
    collection = chroma_client.create_collection("ecoinvent")
    
    # 存储向量
    print("正在存储向量...")
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
    
    print(f"✅ 向量数据库初始化完成！共 {total} 条记录")
    return True

if __name__ == "__main__":
    check_and_init_db()
