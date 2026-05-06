import os
import shutil

# 清理所有可能的缓存位置
cache_paths = [
    os.path.expanduser("~/.cache/huggingface"),
    os.path.expanduser("~/.cache/torch"),
    os.path.expanduser("~/.cache/sentence_transformers"),
    "/root/.cache/huggingface",
    "/root/.cache/torch", 
    "/root/.cache/sentence_transformers",
]

for p in cache_paths:
    if os.path.exists(p):
        try:
            shutil.rmtree(p)
            print(f"已清理: {p}")
        except:
            pass

# 确保不使用镜像
if 'HF_ENDPOINT' in os.environ:
    del os.environ['HF_ENDPOINT']

# 强制重新安装transformers
os.system("pip install --upgrade transformers --force-reinstall --no-cache-dir")

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
