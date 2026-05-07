"""
Carbon Footprint Factor Selector - Streamlit Version
碳足迹因子智能选择器
"""
import os
import sys
import time
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)

# 禁用Streamlit的详细日志
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logging.getLogger("streamlit").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# 记录当前使用的API Key索引
_current_api_idx = 0
_api_lock = logging.getLogger(__name__).handlers[0].lock if logging.getLogger(__name__).handlers else None

# 使用 Hugging Face 中国镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import random
import traceback
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============ 配置 ============
# LLM并发执行器
executor = ThreadPoolExecutor(max_workers=4)

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K = 150
MIN_RESULTS = 6
MAX_RESULTS = 30

# ============ API配置（支持多API轮询）============
DEEPSEEK_API_KEYS = [
    "sk-047bd10a8be74a42a60f881fe3a55e9a",
    "sk-3dfefa654b4a48f991aeda80e610f410",
    "sk-91e55627ff874a5ca95dd5990110b187",
    "sk-2dada24f757c413aac965ba6215a8016",
]
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# API轮询索引
_api_key_index = 0
import threading
_api_lock = threading.Lock()

def get_next_client():
    """轮询获取下一个可用的API客户端，返回(client, api_idx)"""
    global _api_key_index
    with _api_lock:
        api_key = DEEPSEEK_API_KEYS[_api_key_index % len(DEEPSEEK_API_KEYS)]
        _api_key_index += 1
        current_idx = _api_key_index
    # 显示使用的API Key（只显示前8位）
    key_display = api_key[:12] + "..."
    logger.info(f"[API-{current_idx}] ▶ 使用Key: {key_display}")
    # 禁用HTTPX自动重试，避免"Retrying request"日志
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, max_retries=0)
    return client, current_idx

def call_deepseek(messages, temperature=0.3, max_retries=3, timeout=60):
    """带轮询和重试的DeepSeek调用"""
    for attempt in range(max_retries):
        try:
            client, api_idx = get_next_client()
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                temperature=temperature,
                timeout=timeout
            )
            logger.info(f"[API-{api_idx}] ✓ 调用成功")
            return response
        except Exception as e:
            logger.warning(f"[API-{api_idx}] ✗ 失败: {str(e)[:50]}")
            if attempt < max_retries - 1:
                logger.warning(f"    重试中... ({attempt+1}/{max_retries})")
                time.sleep(1)
            else:
                logger.error(f"    API调用彻底失败")
                raise e
    return None

GEOGRAPHIES = [
    "Austria (AT)", "Canada, Quebec (CA-QC)", "Europe without Switzerland and Austria",
    "India (IN)", "Rest-of-World (RoW)", "Switzerland (CH)", "Germany (DE)",
    "USA", "China (CN)", "Japan (JP)", "Brazil (BR)", "Australia (AU)",
    "France (FR)", "United Kingdom (GB)", "GLO"
]

TIME_RANGES = [
    "2012-2025", "2015-2022", "2018-2023", "2020-2025", "2010-2020",
    "2015-2020", "2018-2022", "2021-2026", "2019-2024"
]

collection = None
model = None

@st.cache_resource
def init_models():
    global collection, model
    logger.info("="*50)
    logger.info("🚀 启动碳足迹因子智能选择器")
    logger.info("="*50)
    logger.info("📦 正在加载向量数据库...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection("ecoinvent")
    logger.info(f"✅ 向量数据库加载完成，共 {collection.count()} 条记录")
    
    logger.info("🤖 正在加载Embedding模型...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info(f"✅ Embedding模型加载完成: {EMBEDDING_MODEL}")
    logger.info("="*50)
    logger.info("✨ 系统初始化完成，可以开始检索！")
    logger.info("="*50)
    return collection, model


def extract_keywords(text):
    if not text:
        return []
    prompt = f"""用户输入了一段生产工艺描述：
"{text}"

请提取其中最核心的关键词（2-4个），用于在LCA数据库中检索生产工艺。

要求：
1. 只提取名词性关键词（如：热裂解、流化床、蒸馏）
2. 忽略修饰词（如：利用、进行、这个、一种）
3. 最多4个关键词
4. 输出格式：每行一个关键词，不要解释"""
    try:
        response = call_deepseek([{"role": "user", "content": prompt}], temperature=0.3)
        keywords = response.choices[0].message.content.strip().split('\n')
        return [k.strip() for k in keywords if k.strip()]
    except:
        return [text]

def expand_keywords(text):
    if not text:
        return []
    prompt = f"""用户输入：{text}

请按以下步骤处理：
1. 将用户输入翻译成英文（这作为"完全一致"项）
2. 额外提供4个相关英文同义词或相关术语

关联性定义：
- 完全一致：英文翻译，与原意等价
- 极高：非常接近的英文替代词
- 高：相关性强的英文相关词
- 一般：有一定关联但可能产生噪音的英文词
- 低：关联性较弱的英文词

【关键要求】
- 禁止输出任何中文字符！
- 只输出英文关键词！
- 第一个必须是英文翻译（标记为"完全一致"）
- 后续4个是英文同义词（从"极高"、"高"、"一般"、"低"中选择标注）

输出格式（严格按此格式，每行一个）：
关键词|关联性

例如输入"pp塑料"：
PP plastic|完全一致
polypropylene|极高
polypropylene resin|极高
thermosetting plastic|高
industrial polymer|一般

最多5个，全部必须是英文！"""
    try:
        logger.info("   正在调用API...")
        response = call_deepseek([{"role": "user", "content": prompt}], temperature=0.3, timeout=30)
        logger.info("   API响应已接收，正在解析...")
        result = response.choices[0].message.content.strip()
        keywords_with_level = []
        for line in result.split('\n'):
            line = line.strip()
            if '|' in line:
                parts = line.split('|')
                if len(parts) == 2:
                    keyword = parts[0].strip()
                    level = parts[1].strip()
                    if keyword and level in ["完全一致", "极高", "高", "一般", "低"]:
                        if re.search(r'[a-zA-Z]', keyword) and not contains_chinese(keyword):
                            keywords_with_level.append((keyword, level))
        logger.info(f"   解析完成，找到 {len(keywords_with_level)} 个关键词")
        return keywords_with_level[:5]
    except Exception as e:
        return []

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def is_info_less_activity(product_name, activity_name):
    p = product_name.strip().lower()
    a = activity_name.strip().lower()
    if a == p:
        return True
    if ',' in p:
        before = p[:p.index(',')].strip()
        after = p[p.index(',')+1:].strip()
        for sep in [', ', ',']:
            if a == before + ' production' + sep + after:
                return True
            if a == before + ' construction' + sep + after:
                return True
    if a == p + ' production' or a == p + ' construction':
        return True
    if a == 'market for ' + p:
        return True
    if a == 'market group for ' + p:
        return True
    return False

LEVEL_WEIGHTS = {"完全一致": 0.9, "极高": 0.7, "高": 0.3}
DEPRECATED_LEVELS = ["一般", "低"]

def calculate_retrieval_quota(keywords_with_level, has_original=True, mode="weight", fixed_quota=80):
    if not keywords_with_level:
        return []
    valid_keywords = [(kw, level) for kw, level in keywords_with_level if level not in DEPRECATED_LEVELS]
    is_fallback = False
    if not valid_keywords:
        is_fallback = True
        any_general = [(kw, level) for kw, level in keywords_with_level if level == "一般"]
        any_low = [(kw, level) for kw, level in keywords_with_level if level == "低"]
        if any_general:
            valid_keywords = any_general
        elif any_low:
            valid_keywords = any_low
    results = []
    total_quota = fixed_quota if mode == "fixed" else 80
    if has_original and not is_fallback:
        quota = round(total_quota * 0.2) if mode == "fixed" else round(total_quota / (sum(LEVEL_WEIGHTS.get(level, 0.3) for _, level in valid_keywords) + 1.0))
        results.append((None, quota))
    if is_fallback:
        count = len(valid_keywords)
        quota = round(total_quota / count)
        for keyword, level in valid_keywords:
            results.append((keyword, quota))
    else:
        if mode == "fixed":
            remaining_quota = total_quota - sum(q for _, q in results)
            count = len(valid_keywords)
            if count > 0:
                quota = round(remaining_quota / count)
                for keyword, level in valid_keywords:
                    results.append((keyword, quota))
        else:
            total_weight = sum(LEVEL_WEIGHTS.get(level, 0.3) for _, level in valid_keywords)
            if has_original:
                total_weight += 1.0
            for keyword, level in valid_keywords:
                weight = LEVEL_WEIGHTS.get(level, 0.3)
                quota = round(total_quota * weight / total_weight)
                results.append((keyword, quota))
    return results

def expand_process_keywords(process_text):
    if not process_text:
        return []
    keywords = extract_keywords(process_text)
    expanded = []
    for kw in keywords:
        exp = expand_keywords(kw)
        expanded.extend(exp)
    seen = {}
    for keyword, level in expanded:
        if keyword not in seen:
            seen[keyword] = level
    return list(seen.items())

def judge_waste(judge_text):
    if not judge_text:
        return None
    prompt = f"""判断以下描述是否涉及废弃物处理：
"{judge_text}"
只输出：是 或 否"""
    try:
        response = call_deepseek([{"role": "user", "content": prompt}], temperature=0.1)
        result = response.choices[0].message.content.strip()
        return True if "是" in result else False if "否" in result else None
    except:
        return None

def judge_infrastructure(product_name):
    if not product_name:
        return None
    prompt = f"""判断以下产品名称是否属于"设施/设备"类型：
{product_name}
设施：工厂、发电厂、建筑物、机械设备、交通工具等
不是设施：工业原材料、化学品、零部件等
只输出：是 或 否"""
    try:
        response = call_deepseek([{"role": "user", "content": prompt}], temperature=0.1)
        result = response.choices[0].message.content.strip()
        return True if "是" in result else False if "否" in result else None
    except:
        return None

def build_score_prompt(candidates, user_input):
    """构建打分prompt"""
    context_lines = []
    for j, c in enumerate(candidates):
        product_name = c['product_name']
        activity_desc = c.get('activity_display', '')
        if activity_desc:
            context_lines.append(f"【因子{j+1}】\n产品名: {product_name}\n工艺: {activity_desc}")
        else:
            context_lines.append(f"【因子{j+1}】\n产品名: {product_name}")
    context = "\n".join(context_lines)
    
    return f"""用户需求：{user_input}

候选因子清单（共{len(candidates)}个）：
{context}

请对上述{len(candidates)}个候选因子逐一独立打分（0-100分）。

【关键要求】
- 严格独立评分：评估每个因子时，不参考其他因子的评分上下文
- 产品名匹配度 > 工艺匹配度（如有）
- 再生材料/绿色属性：用户无特殊要求时，该属性减3分

输出格式（每个因子一行，严格按序号）：
1: 85 产品名高度匹配
2: 72 工艺相关
...
{len(candidates)}: 50 相关性较低"""

def score_batch_async(candidates, user_input):
    """构建异步请求"""
    total = len(candidates)
    
    # 分批规则：至多3批，每批至少20条，尽量均分
    if total <= 20:
        num_batches = 1
        batch_size = total
    elif total <= 40:
        num_batches = 2
        batch_size = (total + 1) // 2
    else:
        num_batches = 3
        batch_size = (total + 2) // 3  # 向上取整，尽量均分
    
    batches = []
    batch_starts = []  # 记录每批的起始索引
    for i in range(num_batches):
        start = i * batch_size
        batch_starts.append(start)
        end = start + batch_size if i < num_batches - 1 else total
        batch = candidates[start:end]
        batches.append(batch)
    
    logger.info(f"   分批策略: {num_batches}批")
    for i, b in enumerate(batches):
        logger.info(f"     第{i+1}批: {len(b)}条")
    
    prompts = [build_score_prompt(batch, user_input) for batch in batches]
    
    # 同时推送所有请求，不等待返回
    futures = []
    for i, prompt in enumerate(prompts):
        future = executor.submit(call_deepseek, [{"role": "user", "content": prompt}], temperature=0.3)
        futures.append((i, future))
    
    # 收集结果
    all_scores = {}
    for batch_id, future in futures:
        batch = batches[batch_id]
        start_idx = batch_starts[batch_id]
        try:
            response = future.result()
            result = response.choices[0].message.content.strip()
            
            for j in range(len(batch)):
                match = re.search(rf'^{j+1}:\s*(\d+)(?:\s+(.+))?$', result, re.MULTILINE)
                if match:
                    score = int(match.group(1))
                    reason = match.group(2).strip() if match.group(2) else "无"
                    all_scores[start_idx + j] = {'score': score, 'reason': reason[:50]}
                else:
                    all_scores[start_idx + j] = {'score': 50, 'reason': '解析失败'}
        except Exception as e:
            for j in range(len(batch)):
                all_scores[start_idx + j] = {'score': 50, 'reason': f'异常'}
    
    return all_scores

def score_candidates_with_llm(candidates, user_input):
    """并发批量打分"""
    total = len(candidates)
    
    logger.info(f"   开始LLM打分，共{total}个候选因子")
    
    scores = score_batch_async(candidates, user_input)
    
    logger.info(f"   ✅ LLM打分完成，共{total}个候选因子")
    return scores

def calculate_final_score(llm_score, user_input_is_waste, cut_off_classification, user_is_infra, candidate_is_infra, is_factor_a, user_has_process):
    score = llm_score
    deductions = []
    cut_off_lower = cut_off_classification.lower()
    db_is_product = cut_off_lower == "allocatable product"
    db_is_waste = cut_off_lower == "waste"
    db_is_recyclable = cut_off_lower == "recyclable"
    if user_input_is_waste is not None:
        if user_input_is_waste and db_is_product:
            score -= 25
            deductions.append({"规则": "规则1-废弃物", "判断": f"用户废弃物={user_input_is_waste}, 数据库={cut_off_lower}", "扣分": -25})
        elif not user_input_is_waste and (db_is_waste or db_is_recyclable):
            score -= 25
            deductions.append({"规则": "规则1-废弃物", "判断": f"用户废弃物={user_input_is_waste}, 数据库={cut_off_lower}", "扣分": -25})
    if not user_has_process and not is_factor_a:
        score -= 5
        deductions.append({"规则": "规则2-低信息量activity", "判断": "用户未填工艺+非因子A", "扣分": -5})
    if candidate_is_infra and user_is_infra == False:
        score -= 10
        deductions.append({"规则": "规则3-设施设备", "判断": "用户设施=否, 候选设施=是", "扣分": -10})
    score = max(0, score)
    return score, deductions

def generate_random_geographies():
    num_geos = random.randint(2, 5)
    row_geo = "Rest-of-World (RoW)"
    geos = [row_geo]
    available = [g for g in GEOGRAPHIES if g != row_geo]
    geos.extend(random.sample(available, min(num_geos - 1, len(available))))
    return geos


def search_and_recommend(cas_number, product_name, process, remark):
    if not product_name.strip():
        return None, "请输入产品名称开始匹配"
    
    # 计时器
    start_time = time.time()
    
    # 各环节耗时统计
    timing = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(pct, msg):
        elapsed = time.time() - start_time
        progress_bar.progress(pct)
        status_text.text(f"{msg} ({elapsed:.1f}s)")
    
    def log_timing(name, start):
        elapsed = time.time() - start
        timing[name] = elapsed
        logger.info(f"   ⏱️ {name}: {elapsed:.1f}s")
    
    # ========== CMD日志输出 ==========
    logger.info("")
    logger.info("━"*50)
    logger.info("🔍 开始新的检索任务")
    logger.info("━"*50)
    logger.info(f"📝 输入信息:")
    logger.info(f"   CAS号: {cas_number if cas_number else '未填写'}")
    logger.info(f"   产品名: {product_name}")
    logger.info(f"   工艺: {process if process else '未填写'}")
    logger.info(f"   备注: {remark if remark else '未填写'}")
    logger.info("")
    
    update_progress(0.05, "AI扩展同义词...")
    product_keywords = []
    process_keywords = []
    
    t_product_kw = time.time()
    if product_name:
        logger.info("🔄 正在通过DeepSeek扩展产品关键词...")
        product_keywords = expand_keywords(product_name.strip())
        log_timing("产品关键词扩展", t_product_kw)
        if product_keywords:
            kw_display = ", ".join([f"{kw}({lv})" for kw, lv in product_keywords[:5]])
            logger.info(f"✅ 产品关键词扩展完成: {kw_display}")
        else:
            logger.warning("⚠️ 产品关键词扩展失败，使用原始输入")
    
    t_process_kw = time.time()
    if process:
        logger.info("🔄 正在通过DeepSeek扩展工艺关键词...")
        process_keywords = expand_process_keywords(process.strip())
        log_timing("工艺关键词扩展", t_process_kw)
        if process_keywords:
            kw_display = ", ".join([f"{kw}({lv})" for kw, lv in process_keywords[:5]])
            logger.info(f"✅ 工艺关键词扩展完成: {kw_display}")
        else:
            logger.warning("⚠️ 工艺关键词扩展失败")
    else:
        log_timing("工艺关键词扩展", t_process_kw)
        logger.info("📌 未填写工艺，跳过工艺关键词扩展")
    
    update_progress(0.1, "向量检索中...")
    logger.info("")
    
    # ========== 预判任务：与向量检索并行执行 ==========
    judge_text = " ".join([p for p in [product_name, process, remark] if p])
    future_waste = executor.submit(judge_waste, judge_text)
    future_infra = executor.submit(judge_infrastructure, product_name)
    
    logger.info("📊 开始向量检索...")
    logger.info(f"   候选集大小目标: 70条")
    has_chinese = contains_chinese(product_name)
    product_quota = calculate_retrieval_quota(product_keywords, has_original=not has_chinese, mode="fixed", fixed_quota=55)
    
    all_docs = []
    all_metadatas = []
    all_distances = []
    all_sources = []
    seen_ids = set()
    cas_matched_ids = set()
    
    # CAS精确匹配
    cas_match_count = 0
    if cas_number:
        cas_clean = cas_number.replace(" ", "").replace("-", "").strip()
        if cas_clean:
            logger.info(f"🔍 正在通过CAS号精确匹配: {cas_clean}")
            all_data = collection.get()
            for i, meta in enumerate(all_data['metadatas']):
                db_cas = (meta.get('cas_number', '') or '').replace(" ", "").replace("-", "").strip()
                if db_cas == cas_clean:
                    doc_id = meta.get('activity_name', '') + meta.get('product_name', '')
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(all_data['documents'][i])
                        all_metadatas.append(meta)
                        all_distances.append(0.0)
                        all_sources.append(f"CAS精确:{cas_number.strip()}")
                        cas_matched_ids.add(doc_id)
                        cas_match_count += 1
                        if len(all_docs) >= 30:
                            break
            logger.info(f"✅ CAS精确匹配: 找到 {cas_match_count} 条记录")
    
    # 产品关键词检索
    logger.info("🔍 正在通过产品关键词检索...")
    for keyword, quota in product_quota:
        if keyword is None:
            source_name = f"原始输入:{product_name.strip()}"
            query_emb = model.encode([product_name.strip()])
            search_term = product_name.strip()
        else:
            source_name = f"同义词:{keyword}"
            query_emb = model.encode([keyword])
            search_term = keyword
        results = collection.query(query_embeddings=query_emb.tolist(), n_results=quota)
        added = 0
        for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results.get('distances', [[]])[0]):
            doc_id = meta.get('activity_name', '') + meta.get('product_name', '')
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)
                all_metadatas.append(meta)
                all_distances.append(dist)
                all_sources.append(source_name)
                added += 1
        logger.info(f"   [{search_term}] 新增 {added}/{len(results['documents'][0])} 条")
    
    logger.info(f"📦 产品检索完成，候选集: {len(all_docs)} 条")
    
    # 工艺关键词检索
    if process_keywords:
        logger.info("🔍 正在通过工艺关键词检索...")
        process_quota = calculate_retrieval_quota(process_keywords, has_original=not contains_chinese(process), mode="fixed", fixed_quota=15)
        for keyword, quota in process_quota:
            if keyword is None:
                source_name = f"原始工艺:{process.strip()}"
                query_emb = model.encode([process.strip()])
                search_term = process.strip()
            else:
                source_name = f"工艺同义词:{keyword}"
                query_emb = model.encode([keyword])
                search_term = keyword
            results = collection.query(query_embeddings=query_emb.tolist(), n_results=quota)
            added = 0
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results.get('distances', [[]])[0]):
                doc_id = meta.get('activity_name', '') + meta.get('product_name', '')
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)
                    all_metadatas.append(meta)
                    all_distances.append(dist)
                    all_sources.append(source_name)
                    added += 1
            logger.info(f"   [{search_term}] 新增 {added}/{len(results['documents'][0])} 条")
        
        logger.info(f"📦 工艺检索完成，候选集: {len(all_docs)} 条")
    
    # market for补充
    if not process or not process.strip():
        logger.info("🔍 补充Market for检索...")
        for keyword, quota in product_quota:
            search_term = product_name.strip() if keyword is None else keyword
            market_query = f"market for {search_term}"
            try:
                query_emb = model.encode([market_query])
                results = collection.query(query_embeddings=query_emb.tolist(), n_results=15)
                added = 0
                for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results.get('distances', [[]])[0]):
                    doc_id = meta.get('activity_name', '') + meta.get('product_name', '')
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
                        all_metadatas.append(meta)
                        all_distances.append(dist)
                        all_sources.append(f"Market:{search_term}")
                        added += 1
                if added > 0:
                    logger.info(f"   Market→{search_term}: +{added}")
            except:
                pass
    
    logger.info(f"✅ 向量检索完成，共获取 {len(all_docs)} 条候选")
    
    # 大类因子兜底：当检索质量不佳时补充大类因子
    CATEGORY_FACTOR_IDS = [
        '3544', '3545', '3546', '3547', '3548', '3549', '3550',  # 化学品大类
        '1526', '1527', '1528', '1529',  # 电子元件生产
        '3896', '3897', '3898', '3899',  # 电子元件 market
        '5230', '5234',  # 塑料回收 market
        '7912', '7913',  # 塑料回收生产
    ]
    
    # 触发条件：平均距离 > 0.86 或 结果数量 < 10 且 平均距离 > 0.80
    need_category_fallback = False
    if len(all_distances) > 0:
        avg_distance = sum(all_distances) / len(all_distances)
        max_distance = max(all_distances)
        logger.info(f"   检索质量指标: 平均距离={avg_distance:.3f}, 最大距离={max_distance:.3f}")
        if avg_distance > 0.86 or (len(all_docs) < 10 and avg_distance > 0.80):
            need_category_fallback = True
    elif len(all_docs) < 10:
        need_category_fallback = True
    
    if need_category_fallback:
        logger.info("   🔄 检索质量不佳，触发大类因子兜底...")
        category_data = collection.get(ids=CATEGORY_FACTOR_IDS)
        added = 0
        for i, doc_id in enumerate(category_data['ids']):
            meta = category_data['metadatas'][i]
            doc = category_data['documents'][i]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)
                all_metadatas.append(meta)
                all_distances.append(1.0)  # 大类因子距离设为1.0
                all_sources.append(f"大类兜底:{meta.get('product_name', 'N/A')[:15]}")
                added += 1
        logger.info(f"   大类因子补充完成: +{added} 条")
        logger.info(f"   候选集更新为: {len(all_docs)} 条")
    
    docs = all_docs
    metadatas = all_metadatas
    distances = all_distances
    sources = all_sources
    
    # 电力关键词检测：记录是否包含电力相关关键词
    product_input_lower = product_name.lower()
    user_input_has_electricity = any(kw.lower() in product_input_lower for kw in ['电力', 'electricity'])
    
    # 电力关键词强制候选：若用户输入包含"电力"或"electricity"，强制添加中压电力因子
    if user_input_has_electricity:
        electricity_id = '3881'  # market for electricity, medium voltage
        result = collection.get(ids=[electricity_id])
        if result['documents']:
            # 用与向量检索相同的doc_id格式进行去重
            meta = result['metadatas'][0]
            elec_doc_id = meta.get('activity_name', '') + meta.get('product_name', '')
            if elec_doc_id not in seen_ids:
                logger.info("   ⚡ 检测到电力关键词，强制添加中压电力因子...")
                seen_ids.add(elec_doc_id)
                all_docs.append(result['documents'][0])
                all_metadatas.append(meta)
                all_distances.append(0.0)
                all_sources.append("电力强制")
                logger.info(f"   已添加: market for electricity, medium voltage")
            else:
                logger.info(f"   ⚡ 电力因子已存在于检索结果中，跳过强制添加")
    
    docs = all_docs
    metadatas = all_metadatas
    distances = all_distances
    sources = all_sources
    
    # 输出所有候选因子详情
    logger.info("")
    logger.info(f"📋 候选因子列表 (共{len(metadatas)}条):")
    logger.info("━" * 120)
    logger.info(f"{'#':^4}|{'Product Name':^40}|{'Activity Name':^35}|{'来源':^20}")
    logger.info("━" * 120)
    for idx, (meta, src) in enumerate(zip(metadatas, sources), 1):
        product = meta.get('product_name', 'N/A')[:38]
        activity = meta.get('activity_name', 'N/A')[:33]
        src_short = src[:18]
        logger.info(f"{idx:^4}|{product:<40}|{activity:<35}|{src_short:<20}")
    logger.info("━" * 120)
    
    cas_matched_ids_topk = set()
    for meta in metadatas:
        doc_id = meta.get('activity_name', '') + meta.get('product_name', '')
        if doc_id in cas_matched_ids:
            cas_matched_ids_topk.add(doc_id)
    
    # ========== 等待预判任务完成 ==========
    logger.info("")
    logger.info("⏳ 等待AI预判断完成...")
    t_ai_prejudge = time.time()
    
    user_is_waste = future_waste.result()
    t_infra_judge = time.time()
    user_is_infra = future_infra.result()
    
    logger.info(f"   ✓ 废弃物预判断: {'是' if user_is_waste else '否' if user_is_waste == False else '不确定'} ({t_ai_prejudge - time.time():.1f}s)")
    logger.info(f"   ✓ 设施预判断: {'是' if user_is_infra else '否' if user_is_infra == False else '不确定'} ({t_infra_judge - time.time():.1f}s)")
    
    # ========== P0优化：过滤低质候选 + LLM打分 ==========
    DISTANCE_THRESHOLD = 1.1  # 放宽阈值，让更多候选通过
    
    # 构建候选因子列表（过滤低质量）
    candidates = []
    for i, (meta, dist) in enumerate(zip(metadatas, distances)):
        # P0优化：过滤向量距离>阈值的高噪声候选
        if dist > DISTANCE_THRESHOLD:
            continue
        
        product_name_cand = meta.get('product_name', 'N/A')
        activity_name_cand = meta.get('activity_name', 'N/A')
        if is_info_less_activity(product_name_cand, activity_name_cand):
            activity_display = ""
        else:
            activity_display = activity_name_cand
        candidates.append({
            'idx': i,
            'product_name': product_name_cand,
            'activity_display': activity_display,
            'distance': dist
        })
    
    logger.info(f"   过滤后候选数量: {len(candidates)} 条 (阈值>{DISTANCE_THRESHOLD})")
    
    # 开始LLM打分（预判已完成，不再并行）
    user_full_input = f"产品: {product_name}\n工艺: {process}\n备注: {remark}"
    llm_scores = score_candidates_with_llm(candidates, user_full_input)
    
    update_progress(0.7, "计算最终分数...")
    logger.info("📐 正在计算最终分数并应用规则...")
    scored_results = []
    user_has_process = bool(process and process.strip())
    
    # 构建候选索引映射：candidates中的位置 -> 原始metadatas位置
    candidate_idx_map = {c['idx']: idx for idx, c in enumerate(candidates)}
    
    for i, (doc, meta) in enumerate(zip(docs, metadatas)):
        cut_off = meta.get('cut_off', '')
        candidate_sector = meta.get('sector', '')
        candidate_is_infra = "infrastructure" in candidate_sector.lower() and "machinery" in candidate_sector.lower()
        doc_id = meta.get('activity_name', '') + meta.get('product_name', '')
        is_cas_matched = doc_id in cas_matched_ids_topk
        candidate_product = meta.get('product_name', '')
        candidate_activity = meta.get('activity_name', '')
        is_factor_a = is_info_less_activity(candidate_product, candidate_activity)
        
        # 从过滤后的candidates中获取分数
        cand_pos = candidate_idx_map.get(i)
        if cand_pos is None:
            continue  # 距离超阈值，不打分，不显示
        
        score_data = llm_scores.get(cand_pos)
        if score_data is None:
            continue  # LLM未打分，不显示
        
        llm_base_score = score_data['score']
        llm_reason = score_data['reason']
        if is_cas_matched and llm_base_score < 85:
            llm_base_score = 85
            llm_reason = "CAS精确匹配保底85分"
        final_score, deductions = calculate_final_score(llm_base_score, user_is_waste, cut_off, user_is_infra, candidate_is_infra, is_factor_a, user_has_process)
        scored_results.append({
            'meta': meta,
            'score': final_score,
            'llm_score': llm_base_score,
            'llm_reason': llm_reason,
            'deductions': deductions,
            'is_waste': cut_off.lower() in ["waste", "recyclable"],
            'is_cas_matched': is_cas_matched,
            'geographies': generate_random_geographies(),
            'time_period': random.choice(TIME_RANGES)
        })
    
    # 电力因子加分：仅当用户输入包含电力关键词时，中压电力因子加6分，不超过99分
    if user_input_has_electricity:
        for r in scored_results:
            if r['meta'].get('product_name', '') == 'electricity, medium voltage':
                r['score'] = min(r['score'] + 6, 99)
                r['llm_reason'] = (r['llm_reason'] or '') + ' | 电力因子+6分'
    
    # 多样性调整：相同product_name的因子，重复出现时逐步扣排序分
    # 策略：先按原始分数降序排列，高分先遇到
    # 惩罚规则：第2次0分，第3次-5，第4次-15，第5次+-25
    scored_results.sort(key=lambda x: -x['score'])  # 按原始分数降序
    
    product_count = {}  # 记录每个product_name的处理次序
    for r in scored_results:
        product_name = r['meta'].get('product_name', 'N/A')
        count = product_count.get(product_name, 0) + 1
        product_count[product_name] = count
        # 计算排序分：原始分数 - 重复扣分
        if count == 1:
            sort_penalty = 0
        elif count == 2:
            sort_penalty = 0
        elif count == 3:
            sort_penalty = -5
        elif count == 4:
            sort_penalty = -15
        else:
            sort_penalty = -25
        r['sort_score'] = r['score'] + sort_penalty
    
    # 按排序分排序（考虑多样性调整）
    scored_results.sort(key=lambda x: (-x['sort_score'], x.get('distance', 0)))
    
    # 过滤和截取结果（基于原始分数，不是排序分）
    filtered_results = [r for r in scored_results if r['score'] >= 60]
    top_results = filtered_results[:MAX_RESULTS] if len(filtered_results) >= MIN_RESULTS else scored_results[:MIN_RESULTS]
    
    # 输出完整LLM打分结果
    logger.info("")
    logger.info(f"📊 LLM打分结果 (共{len(scored_results)}条):")
    logger.info("━" * 140)
    logger.info(f"{'#':^4}|{'Product Name':^35}|{'Activity Name':^30}|{'LLM分':^6}|{'终分':^6}|{'状态'}")
    logger.info("━" * 140)
    for i, r in enumerate(scored_results, 1):
        product = r['meta'].get('product_name', 'N/A')[:33]
        activity = r['meta'].get('activity_name', 'N/A')[:28]
        llm_score = r['llm_score']
        final_score = r['score']
        status = '★CAS' if r.get('is_cas_matched') else ('✓优' if final_score>=90 else ('✓良' if final_score>=80 else ('△中' if final_score>=60 else '✗')))
        logger.info(f"{i:^4}|{product:<35}|{activity:<30}|{llm_score:^6}|{final_score:^6}|{status}")
    logger.info("━" * 140)
    
    # 输出统计信息
    logger.info("")
    logger.info("📈 检索结果统计:")
    logger.info(f"   总候选数: {len(scored_results)} 条")
    logger.info(f"   ≥60分(通过筛选): {len(filtered_results)} 条")
    logger.info(f"   ≥90分(高分推荐): {sum(1 for r in scored_results if r['score'] >= 90)} 条")
    logger.info(f"   80-89分(良好): {sum(1 for r in scored_results if 80 <= r['score'] < 90)} 条")
    logger.info(f"   60-79分(一般): {sum(1 for r in scored_results if 60 <= r['score'] < 80)} 条")
    logger.info(f"   最终输出: {len(top_results)} 条")
    logger.info(f"   CAS精确匹配: {sum(1 for r in top_results if r.get('is_cas_matched'))} 条")
    logger.info(f"   废弃物类型: {sum(1 for r in top_results if r.get('is_waste'))} 条")
    
    total_time = time.time() - start_time
    update_progress(1.0, f"完成！总用时: {total_time:.1f}秒")
    
    # 输出耗时汇总
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 AI耗时汇总")
    logger.info("=" * 60)
    ai_total = 0
    for name, elapsed in timing.items():
        pct = (elapsed / total_time * 100) if total_time > 0 else 0
        logger.info(f"   {name:<20}: {elapsed:>6.1f}s  ({pct:>5.1f}%)")
        ai_total += elapsed
    logger.info("-" * 60)
    logger.info(f"   {'AI总耗时':<20}: {ai_total:>6.1f}s")
    logger.info(f"   {'非AI耗时':<20}: {total_time - ai_total:>6.1f}s")
    logger.info(f"   {'总耗时':<20}: {total_time:>6.1f}s")
    logger.info("=" * 60)
    
    logger.info("")
    logger.info("━"*50)
    logger.info(f"✅ 检索任务完成！总用时: {total_time:.1f}秒")
    logger.info("━"*50)
    
    # 格式化关键词
    product_kw_str = ", ".join([f"{kw}({lv})" for kw, lv in product_keywords[:5]]) if product_keywords else "无"
    process_kw_str = ", ".join([f"{kw}({lv})" for kw, lv in process_keywords]) if process_keywords else "无"
    
    return top_results, {
        'user_is_waste': user_is_waste,
        'user_is_infra': user_is_infra,
        'product_kw_str': product_kw_str,
        'process_kw_str': process_kw_str,
        'total_time': total_time
    }


# ============ Streamlit UI ============
st.set_page_config(
    page_title="Carbon Footprint Factor Selector",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义CSS
st.markdown("""
<style>
    /* 全局样式 */
    .stApp {
        background: linear-gradient(135deg, #F5F5F5 0%, #f8fafc 100%);
    }
    
    /* 缩小顶部空白 - 强制覆盖 */
    section.main {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    div.block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    
    /* 左侧面板 - 悬浮固定 */
    .left-panel {
        position: sticky;
        top: 5px;
        max-height: calc(100vh - 10px);
        overflow-y: auto;
    }
    
    /* 左侧列容器 */
    [data-testid="stHorizontalBlock"]:first-child > div:first-child {
        position: sticky;
        top: 5px;
        align-self: flex-start;
    }
    
    /* 标题卡片 - 白色背景 */
    .header-card {
        background: white;
        border-radius: 12px;
        border: 2px solid #2EA266;
        box-shadow: 0 4px 12px rgba(46, 162, 102, 0.2);
        padding: 10px 16px;
        margin-bottom: 8px;
    }
    .header-card h1 {
        font-size: 16px;
        font-weight: 700;
        color: #2EA266;
        margin: 0 0 2px 0;
    }
    .header-card p {
        font-size: 11px;
        color: #2EA266;
        margin: 0;
    }
    
    /* 表单卡片 - 白色背景 */
    .form-card {
        background: white;
        border-radius: 12px;
        border: 2px solid #2EA266;
        box-shadow: 0 4px 12px rgba(46, 162, 102, 0.2);
        overflow: hidden;
    }
    
    /* 查询条件标签 */
    .query-label {
        padding: 8px 16px;
        border-bottom: 1px solid #E0E0E0;
        font-size: 13px;
        font-weight: 600;
        color: #2EA266;
    }
    
    /* 表单区域 */
    .form-area {
        padding: 12px 16px;
    }
    
    /* 输入框样式 - 白色背景 */
    .form-area div[data-testid="stTextInput"] > div > div > input,
    .form-area div[data-testid="stTextArea"] > div > div > textarea {
        background-color: white !important;
        border: 1px solid #D0D0D0 !important;
        border-radius: 6px;
    }
    .form-area div[data-testid="stTextInput"] > div > div > input:focus,
    .form-area div[data-testid="stTextArea"] > div > div > textarea:focus {
        border-color: #2EA266 !important;
        box-shadow: 0 0 0 2px rgba(46, 162, 102, 0.2) !important;
    }
    
    /* 增大输入框间距 */
    .form-area div[data-testid="stTextInput"],
    .form-area div[data-testid="stTextArea"] {
        margin-bottom: 12px !important;
    }
    
    /* Search按钮样式 - 绿色背景 */
    div[data-testid="stFormSubmitButton"] button,
    .form-area div[data-testid="stFormSubmitButton"] button,
    div.stButton > button {
        background-color: #2EA266 !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover,
    .form-area div[data-testid="stFormSubmitButton"] button:hover,
    div.stButton > button:hover {
        background-color: #248c4d !important;
    }
    
    /* 提示区 */
    .hint-box {
        padding: 8px 16px;
        background: #F5F5F5;
    }
    .hint-box p {
        font-size: 12px;
        color: #2EA266;
        line-height: 1.4;
        padding: 8px 12px;
        background: white;
        border: 1px solid #2EA266;
        border-radius: 6px;
        margin: 0;
    }
    
    /* 结果卡片 */
    .result-card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #2EA266;
        padding: 20px 24px;
        margin-bottom: 16px;
        transition: all 0.2s;
    }
    .result-card:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    
    /* 分数圆圈 */
    .score-circle {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        font-weight: 700;
    }
    .score-high { background: #E0E0E0; border: 3px solid #2EA266; color: #2EA266; }
    .score-medium { background: #FFF9C4; border: 3px solid #F9A825; color: #F57F17; }
    .score-low { background: #FFCDD2; border: 3px solid #D32F2F; color: #B71C1C; }
    
    /* 统计卡片 */
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #2EA266;
        text-align: center;
    }
    .stat-num {
        font-size: 32px;
        font-weight: 700;
        color: #0f172a;
    }
    .stat-label {
        font-size: 13px;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* 关键词标签 */
    .kw-tag {
        display: inline-block;
        padding: 4px 10px;
        background: #F5F5F5;
        border: 1px solid #D0D0D0;
        border-radius: 4px;
        font-size: 12px;
        color: #2EA266;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    
    /* 地理位置 */
    .geo-tag {
        display: inline-block;
        padding: 3px 8px;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        font-size: 11px;
        color: #475569;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    
    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def main():
    # 初始化模型
    global collection, model
    if collection is None or model is None:
        collection, model = init_models()
    
    # 主布局：左侧输入 + 右侧结果
    col_left, col_right = st.columns([1, 2.5], gap="large")
    
    with col_left:
        # 卡片1：标题
        st.markdown('''
        <div class="header-card">
            <h1>Carbon Footprint Factor Selector</h1>
            <p>智能碳足迹因子选择器</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # 卡片2：查询表单
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        
        with st.form("search_form"):
            st.markdown('<div class="query-label">查询条件</div>', unsafe_allow_html=True)
            st.markdown('<div class="form-area">', unsafe_allow_html=True)
            
            st.markdown('<div style="font-size:13px;font-weight:500;color:#374151;margin-bottom:6px;">CAS号</div>', unsafe_allow_html=True)
            cas_input = st.text_input("CAS号", placeholder="如：7439-89-6（可选）", label_visibility="collapsed")
            st.markdown('<div style="font-size:13px;font-weight:500;color:#374151;margin-bottom:6px;">名称 *</div>', unsafe_allow_html=True)
            product_input = st.text_input("名称", placeholder="如：生铁、碳素钢", label_visibility="collapsed")
            st.markdown('<div style="font-size:13px;font-weight:500;color:#374151;margin-bottom:6px;">工艺/路线</div>', unsafe_allow_html=True)
            process_input = st.text_input("工艺/路线", placeholder="如：转炉炼钢（可选）", label_visibility="collapsed")
            st.markdown('<div style="font-size:13px;font-weight:500;color:#374151;margin-bottom:6px;">备注</div>', unsafe_allow_html=True)
            remark_input = st.text_area("备注", placeholder="用途说明（可选）", label_visibility="collapsed", height=60)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            search_btn = st.form_submit_button("🔍 Search", use_container_width=True)
            
            st.markdown('<div class="hint-box"><p>💡 填写产品名称后点击 Search，系统返回评分≥60分的候选因子（最少6条，最多30条）</p></div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # 关闭form-card
    
    with col_right:
        if search_btn and product_input:
            with st.spinner("正在检索..."):
                top_results, meta_info = search_and_recommend(
                    cas_input, product_input, process_input, remark_input
                )
            
            if top_results:
                # 统计信息
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.markdown(f'''
                    <div class="stat-card">
                        <div class="stat-num">{len(top_results)}</div>
                        <div class="stat-label">匹配因子</div>
                    </div>
                    ''', unsafe_allow_html=True)
                with stat_cols[1]:
                    st.markdown(f'''
                    <div class="stat-card">
                        <div class="stat-num">{sum(1 for r in top_results if r['score'] >= 90)}</div>
                        <div class="stat-label">高分推荐</div>
                    </div>
                    ''', unsafe_allow_html=True)
                with stat_cols[2]:
                    st.markdown(f'''
                    <div class="stat-card">
                        <div class="stat-num">{sum(1 for r in top_results if not r['deductions'])}</div>
                        <div class="stat-label">完全匹配</div>
                    </div>
                    ''', unsafe_allow_html=True)
                with stat_cols[3]:
                    total_time = meta_info.get('total_time', 0)
                    st.markdown(f'''
                    <div class="stat-card" style="border-left-color:#3b82f6;">
                        <div class="stat-num" style="color:#3b82f6;">{total_time:.1f}s</div>
                        <div class="stat-label">总用时</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # 筛选状态
                waste_str = "是" if meta_info['user_is_waste'] == True else ("否" if meta_info['user_is_waste'] == False else "不确定")
                infra_str = "是" if meta_info['user_is_infra'] == True else ("否" if meta_info['user_is_infra'] == False else "不确定")
                
                st.markdown(f'''
                <div style="background:white;padding:12px 20px;border-radius:8px;margin-bottom:16px;display:flex;gap:24px;">
                    <span><strong>废弃物判断：</strong><span style="color:{'#D32F2F' if meta_info['user_is_waste'] == True else ('#2EA266' if meta_info['user_is_waste'] == False else '#94a3b8')}">{waste_str}</span></span>
                    <span><strong>设施/设备判断：</strong><span style="color:{'#D32F2F' if meta_info['user_is_infra'] == True else ('#2EA266' if meta_info['user_is_infra'] == False else '#94a3b8')}">{infra_str}</span></span>
                </div>
                ''', unsafe_allow_html=True)
                
                # 关键词
                st.markdown(f'''
                <div style="background:white;padding:16px 20px;border-radius:8px;margin-bottom:16px;">
                    <div style="font-size:12px;color:#94a3b8;margin-bottom:8px;text-transform:uppercase;">检索关键词</div>
                    <span class="kw-tag">产品: {meta_info['product_kw_str'] if meta_info['product_kw_str'] != '无' else '-'}</span>
                    <span class="kw-tag" style="background:#FFF3E0;border-color:#FFCC80;color:#E65100;">工艺: {meta_info['process_kw_str'] if meta_info['process_kw_str'] != '无' else '-'}</span>
                </div>
                ''', unsafe_allow_html=True)
                
                # 结果卡片
                for i, item in enumerate(top_results, 1):
                    meta = item['meta']
                    score = item['score']
                    llm_score = item.get('llm_score', score)
                    llm_reason = item.get('llm_reason', '')
                    deductions = item['deductions']
                    geos = item['geographies']
                    
                    if score >= 90:
                        score_class = "score-high"
                        status_text = "优秀"
                    elif score >= 80:
                        score_class = "score-high"
                        status_text = "良好"
                    elif score >= 60:
                        score_class = "score-medium"
                        status_text = "一般"
                    else:
                        score_class = "score-low"
                        status_text = "需审核"
                    
                    product_name = meta.get('product_name', 'N/A')
                    activity_name = meta.get('activity_name', 'N/A')
                    sector = meta.get('sector', 'N/A')
                    cut_off = meta.get('cut_off', 'N/A')
                    special_activity = meta.get('special_activity', 'N/A')
                    cas_number_db = meta.get('cas_number', '')
                    time_period = item['time_period']
                    
                    deduct_str = ""
                    for d in deductions:
                        if "规则1" in d["规则"]:
                            deduct_str += f'<span style="background:#FFCDD2;color:#B71C1C;padding:2px 6px;border-radius:4px;font-size:11px;margin-right:4px;">规则1 {d["扣分"]}</span>'
                        elif "规则3" in d["规则"]:
                            deduct_str += f'<span style="background:#FFCDD2;color:#B71C1C;padding:2px 6px;border-radius:4px;font-size:11px;margin-right:4px;">规则3 {d["扣分"]}</span>'
                    
                    geo_html = "".join([f'<span class="geo-tag">{g}</span>' for g in geos])
                    
                    st.markdown(f'''
                    <div class="result-card">
                        <div style="display:flex;justify-content:space-between;align-items:start;">
                            <div style="flex:1;">
                                <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                                    <span style="background:#2EA266;color:white;padding:4px 10px;border-radius:6px;font-size:12px;font-weight:600;">#{i}</span>
                                    <span style="background:#FFF9C4;padding:4px 10px;border-radius:6px;font-size:11px;color:#E65100;">
                                        LLM评分: {llm_score}分 {f'| {llm_reason}' if llm_reason else ''}
                                    </span>
                                    {deduct_str if deduct_str else '<span style="background:#E0E0E0;color:#2EA266;padding:2px 6px;border-radius:4px;font-size:11px;">无扣分</span>'}
                                </div>
                                <div style="display:flex;align-items:center;margin-bottom:4px;">
                                    <span style="background:#3b82f6;color:white;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;">Product</span>
                                    <h3 style="font-size:16px;font-weight:600;color:#0f172a;margin:0 0 0 8px;">{product_name}</h3>
                                </div>
                                <div style="display:flex;align-items:center;margin-bottom:16px;">
                                    <span style="background:#3b82f6;color:white;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;">Activity</span>
                                    <p style="font-size:14px;color:#475569;margin:0 0 0 8px;">{activity_name}</p>
                                </div>
                                <div style="display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:#334155;">
                                    <span><strong>Special:</strong> {special_activity}</span>
                                    <span><strong>Period:</strong> {time_period}</span>
                                    <span><strong>Cut-Off:</strong> {cut_off}</span>
                                    <span><strong>CAS:</strong> {cas_number_db if cas_number_db else '-'}</span>
                                </div>
                                <span style="display:inline-block;margin-top:12px;padding:4px 10px;background:#f1f5f9;color:#64748b;border-radius:6px;font-size:12px;">{sector}</span>
                            </div>
                            <div style="text-align:center;min-width:80px;">
                                <div class="score-circle {score_class}">
                                    <span style="font-size:18px;">{score}</span>
                                </div>
                                <span style="font-size:12px;font-weight:600;color:#64748b;margin-top:4px;display:block;">{status_text}</span>
                            </div>
                        </div>
                        <div style="margin-top:16px;padding-top:12px;border-top:1px solid #f1f5f9;">
                            <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;margin-bottom:8px;">Geography</div>
                            {geo_html}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.warning("未找到匹配结果，请尝试其他关键词")
        else:
            st.markdown('''
            <div style="background:white;border-radius:12px;padding:80px 40px;text-align:center;color:#9ca3af;">
                <p style="font-size:16px;">输入产品名称开始检索</p>
            </div>
            ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
