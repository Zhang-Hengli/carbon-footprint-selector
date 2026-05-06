import os
# 使用 Hugging Face 中国镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import gradio as gr
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import random
import traceback
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K = 150
MIN_RESULTS = 6  # 最少显示条数
MAX_RESULTS = 30  # 最多显示条数

# DeepSeek API配置
DEEPSEEK_API_KEY = "sk-047bd10a8be74a42a60f881fe3a55e9a"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 初始化DeepSeek客户端
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# 常见地理位置列表
GEOGRAPHIES = [
    "Austria (AT)", "Canada, Quebec (CA-QC)", "Europe without Switzerland and Austria",
    "India (IN)", "Rest-of-World (RoW)", "Switzerland (CH)", "Germany (DE)",
    "USA", "China (CN)", "Japan (JP)", "Brazil (BR)", "Australia (AU)",
    "France (FR)", "United Kingdom (GB)", "GLO"
]

# 时间范围
TIME_RANGES = [
    "2012-2025", "2015-2022", "2018-2023", "2020-2025", "2010-2020",
    "2015-2020", "2018-2022", "2021-2026", "2019-2024"
]

collection = None
model = None

def init_models():
    global collection, model
    print("正在加载向量数据库...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # 检查数据库是否存在
    try:
        collection = chroma_client.get_collection("ecoinvent")
        print("✓ 数据库已存在")
    except:
        print("✗ 数据库不存在，请先在本地运行 init_database.py 初始化数据库")
        raise Exception("数据库未初始化")
    
    print("正在加载Embedding模型...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("初始化完成！")

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
        response = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        keywords = response.choices[0].message.content.strip().split('\n')
        return [k.strip() for k in keywords if k.strip()]
    except:
        return [text]

def expand_keywords(text):
    """AI扩展同义词并标注关联性等级
    
    返回格式: [(关键词, 关联等级), ...]
    关联等级: "完全一致", "极高", "高", "一般", "低"
    重要：所有输出必须是英文，中文关键词会被过滤掉
    """
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
        response = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        result = response.choices[0].message.content.strip()
        
        keywords_with_level = []
        for line in result.split('\n'):
            line = line.strip()
            if '|' in line:
                parts = line.split('|')
                if len(parts) == 2:
                    keyword = parts[0].strip()
                    level = parts[1].strip()
                    # 过滤中文关键词：必须包含英文字母且不含中文字符
                    if keyword and level in ["完全一致", "极高", "高", "一般", "低"]:
                        if re.search(r'[a-zA-Z]', keyword) and not contains_chinese(keyword):
                            keywords_with_level.append((keyword, level))
        
        return keywords_with_level[:5]
    except Exception as e:
        return []


def contains_chinese(text):
    """检测文本是否包含中文"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def is_info_less_activity(product_name, activity_name):
    """
    判断 activity_name 是否为低信息量（可隐去）
    
    低信息量模式：
    1. 完全相同：activity_name == product_name
    2. production/construction 插入型：A,B -> A production,B 或 A construction,B
    3. production/construction 追加型：product_name -> product_name production/construction
    4. market for 型：product_name -> market for product_name
    """
    p = product_name.strip().lower()
    a = activity_name.strip().lower()
    
    # 1. 完全相同
    if a == p:
        return True
    
    # 2. production/construction 插入或追加模式
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
    
    # 3. market for / market group for 模式
    if a == 'market for ' + p:
        return True
    if a == 'market group for ' + p:
        return True
    
    return False


# 关联等级权重（"一般"和"低"被弃用，不参与检索）
LEVEL_WEIGHTS = {
    "完全一致": 0.9,
    "极高": 0.7,
    "高": 0.3
}

# 被弃用的关联等级（不参与向量检索）
DEPRECATED_LEVELS = ["一般", "低"]


def calculate_retrieval_quota(keywords_with_level, has_original=True, mode="weight", fixed_quota=80):
    """根据关联性等级计算每个关键词的检索配额
    
    has_original: 原始输入是否保留（英文输入为True，中文输入为False）
    mode: 配额模式 - "weight"(按权重分配) 或 "fixed"(固定配额)
    fixed_quota: 当mode="fixed"时使用的总配额（默认80）
    返回: [(关键词, 配额), ...]
    注意: "一般"和"低"等级的关键词被弃用
    极端情况：如果弃用后无关键词，则按以下规则处理：
    - 若有"一般"标签的关键词，则全部"一般"进入检索，配额平均分配
    - 若无"一般"但有"低"标签的关键词，则全部"低"进入检索，配额平均分配
    """
    if not keywords_with_level:
        return []
    
    # 过滤掉被弃用的等级
    valid_keywords = [(kw, level) for kw, level in keywords_with_level if level not in DEPRECATED_LEVELS]
    
    # 标记是否为极端情况（全是一般或低）
    is_fallback = False
    
    # 极端情况：所有关键词都是"一般"或"低"
    if not valid_keywords:
        is_fallback = True
        # 优先使用"一般"，其次使用"低"
        any_general = [(kw, level) for kw, level in keywords_with_level if level == "一般"]
        any_low = [(kw, level) for kw, level in keywords_with_level if level == "低"]
        
        if any_general:
            valid_keywords = any_general
        elif any_low:
            valid_keywords = any_low
    
    results = []
    # 确定使用的总配额
    total_quota = fixed_quota if mode == "fixed" else 80
    
    # 原始输入配额（正常情况才分配原始输入配额）
    if has_original and not is_fallback:
        if mode == "fixed":
            # 固定模式：原始输入占20%
            quota = round(total_quota * 0.2)
        else:
            # 权重模式：按权重分配
            weight_total = sum(LEVEL_WEIGHTS.get(level, 0.3) for _, level in valid_keywords) + 1.0
            quota = round(total_quota / weight_total)
        results.append((None, quota))  # None表示原始输入
    
    # 扩展关键词配额
    if is_fallback:
        # 极端情况：一般或低等级，平均分配配额
        count = len(valid_keywords)
        quota = round(total_quota / count)
        for keyword, level in valid_keywords:
            results.append((keyword, quota))
    else:
        if mode == "fixed":
            # 固定模式：剩余配额平均分配给有效关键词
            remaining_quota = total_quota - sum(q for _, q in results)
            count = len(valid_keywords)
            if count > 0:
                quota = round(remaining_quota / count)
                for keyword, level in valid_keywords:
                    results.append((keyword, quota))
        else:
            # 正常情况：按权重分配配额
            total_weight = sum(LEVEL_WEIGHTS.get(level, 0.3) for _, level in valid_keywords)
            if has_original:
                total_weight += 1.0
            for keyword, level in valid_keywords:
                weight = LEVEL_WEIGHTS.get(level, 0.3)
                quota = round(total_quota * weight / total_weight)
                results.append((keyword, quota))
    
    return results

def expand_process_keywords(process_text):
    """工艺关键词扩展，返回关联性标注"""
    if not process_text:
        return []
    keywords = extract_keywords(process_text)
    expanded = []
    for kw in keywords:
        exp = expand_keywords(kw)  # 返回 [(keyword, level), ...]
        expanded.extend(exp)
    # 去重，保留第一个出现的等级
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
        response = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
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
        response = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        result = response.choices[0].message.content.strip()
        return True if "是" in result else False if "否" in result else None
    except:
        return None


def llm_evaluate_candidates(user_input, docs, metadatas):
    """
    LLM全量分析所有候选因子，返回每个的匹配度分数和理由
    返回格式: {doc_id: {'score': score, 'reason': reason}}
    
    分组并行评分：每组约10条候选因子，各组并行调用去除组间上下文干扰
    """
    if not docs:
        return {}
    
    GROUP_SIZE = 25
    
    # 构造候选因子列表（保留原始索引）
    candidates = []
    total_count = len(docs)
    for i, meta in enumerate(metadatas):
        product_name = meta.get('product_name', 'N/A')
        activity_name = meta.get('activity_name', 'N/A')
        
        # 如果 activity_name 信息量低，置空以节省token
        if is_info_less_activity(product_name, activity_name):
            activity_display = ""
        else:
            activity_display = activity_name
        
        candidates.append({
            'idx': i,
            'product_name': product_name,
            'activity_display': activity_display
        })
    
    # 分组
    groups = [candidates[i:i+GROUP_SIZE] for i in range(0, len(candidates), GROUP_SIZE)]
    print(f"   → LLM并行分析 {total_count} 个候选因子，分 {len(groups)} 组 (每组最多{GROUP_SIZE}条)")
    
    def call_llm(group, group_id):
        """对一组候选因子发起LLM评分"""
        # 构建组内prompt
        context_lines = []
        for j, c in enumerate(group):
            context_lines.append(f"""【候选{j+1}】
产品名称: {c['product_name']}
活动名称: {c['activity_display']}""")
        context = "\n\n".join(context_lines)
        group_size = len(group)
        
        prompt = f"""用户需求：{user_input}

候选因子清单（第{group_id+1}组）：
{context}

请根据用户需求，对上述每个候选因子打分（0-100分），并给出简要理由。

【重要规则】
1. 必须输出{group_size}个评分，一个都不能少！
2. 理由要简洁，不超过20个字
3. 每个评分单独一行，按候选序号排列
4. 请独立评分，不受当前组内其他候选因子的影响，每条因子分数应独立判断
6. 【再生材料/绿色属性匹配规则】
  - 首先判断用户需求中是否明确标注或暗示了以下任何一项：
    * 是再生材料（如：再生、回收、recycled）
    * 来自于生物质即原材料是生物质（如：生物质、biomass、bio-based）
    * 使用了绿色能源（如：绿色能源、renewable、solar、wind、hydroelectric、清洁能源）
  - 如果用户需求中上述三项都没有明确标注或暗示，则：
    * 对于候选因子中明确标注或暗示了上述绿色属性的，应在原有评分基础上减3分
    * 例如：用户输入"碳钢"，候选因子是"steel, carbon, 100% scrap"，应扣3分（因为暗示是再生材料）
    * 例如：用户输入"电力"，候选因子是"electricity, from renewable energy"，应扣3分
  - 如果用户需求中已明确标注或暗示了上述绿色属性，则不适用此扣分规则

输出格式（严格按此格式）：
1: 85 理由：产品名称高度匹配
2: 72 理由：工艺相关但材质有差异
...
{group_size}: 40 理由：相关性较低"""
        
        try:
            print(f"   → 第{group_id+1}组 ({group_size}条) 发送中...")
            response = deepseek_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            result = response.choices[0].message.content.strip()
            
            # 解析
            scores = {}
            for j in range(group_size):
                match = re.search(rf'^{j+1}:\s*(\d+)\s*理由[：:]\s*(.+?)(?=\n\d+:|$)', result, re.MULTILINE | re.DOTALL)
                if match:
                    scores[j] = {
                        'score': int(match.group(1)),
                        'reason': match.group(2).strip()[:50]
                    }
                else:
                    match_simple = re.search(rf'^{j+1}:\s*(\d+)', result, re.MULTILINE)
                    if match_simple:
                        scores[j] = {
                            'score': int(match_simple.group(1)),
                            'reason': '（无理由）'
                        }
                    else:
                        scores[j] = {'score': 50, 'reason': '（解析失败，默认50分）'}
            
            print(f"   → 第{group_id+1}组 完成 ({len(scores)}条)")
            return scores, group
        except Exception as e:
            print(f"   → 第{group_id+1}组 失败: {e}")
            return {j: {'score': 50, 'reason': '（异常，默认50分）'} for j in range(len(group))}, group
    
    # 并行发送所有组
    all_scores = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(call_llm, g, gi): gi for gi, g in enumerate(groups)}
        for future in as_completed(futures):
            scores, group = future.result()
            for local_idx, score_data in scores.items():
                global_idx = group[local_idx]['idx']
                all_scores[global_idx] = score_data
    
    print(f"   → 并行分析完成，共 {len(all_scores)} 条")
    return all_scores

def calculate_final_score(llm_score, user_input_is_waste, cut_off_classification, user_is_infra, candidate_is_infra, is_factor_a, user_has_process):
    """
    在LLM匹配分数基础上应用规则扣分
    
    llm_score: LLM评估的匹配度分数 (0-100)
    规则1(废弃物): -25分
    规则2(低信息量activity): -6分（仅当用户未填写生产工艺时适用）
    规则3(设施设备): -10分
    """
    score = llm_score
    deductions = []
    
    cut_off_lower = cut_off_classification.lower()
    db_is_product = cut_off_lower == "allocatable product"
    db_is_waste = cut_off_lower == "waste"
    db_is_recyclable = cut_off_lower == "recyclable"
    
    user_waste_str = "是" if user_input_is_waste else ("否" if user_input_is_waste == False else "不确定")
    db_cutoff_str = cut_off_lower if cut_off_lower else "空"
    
    if user_input_is_waste is not None:
        if user_input_is_waste and db_is_product:
            score -= 25
            deductions.append({"规则": "规则1-废弃物", "判断": f"用户废弃物={user_waste_str}, 数据库={db_cutoff_str}", "扣分": -25})
        elif not user_input_is_waste and (db_is_waste or db_is_recyclable):
            score -= 25
            deductions.append({"规则": "规则1-废弃物", "判断": f"用户废弃物={user_waste_str}, 数据库={db_cutoff_str}", "扣分": -25})
    
    # 规则2：低信息量activity扣分（仅当用户未填写生产工艺时适用）
    # 因子A（activity name是低信息量的）不扣分，非因子A扣6分
    if not user_has_process:
        if not is_factor_a:
            score -= 6
            deductions.append({"规则": "规则2-低信息量activity", "判断": "用户未填工艺+非因子A", "扣分": -6})
    
    # 规则3：设施设备扣分
    if candidate_is_infra:
        if user_is_infra == False:
            score -= 10
            deductions.append({"规则": "规则3-设施设备", "判断": f"用户设施=否, 候选设施=是", "扣分": -10})
    
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
        return "<div style='padding: 60px; text-align: center; color: #9ca3af; font-size: 16px;'>请输入产品名称开始匹配</div>"
    
    print("\n" + "=" * 50)
    print("🔍 智能碳足迹因子检索")
    print("=" * 50)
    
    # ========== 步骤1: AI扩展关键词并标注关联性 ==========
    product_keywords = []  # [(keyword, level), ...]
    process_keywords = []  # [(keyword, level), ...]
    
    if product_name:
        print(f"📝 产品名称: {product_name}")
        print("⏳ AI扩展同义词并标注关联性...")
        product_keywords = expand_keywords(product_name.strip())
        if product_keywords:
            kw_display = ", ".join([f"{kw}({lv})" for kw, lv in product_keywords[:5]])
            print(f"   → {kw_display}")
        else:
            print("   → 扩展失败")
    
    if process:
        print(f"📝 生产工艺: {process}")
        print("⏳ AI扩展工艺关键词...")
        process_keywords = expand_process_keywords(process.strip())
        if process_keywords:
            # 显示所有工艺同义词（不限制数量）
            kw_display = ", ".join([f"{kw}({lv})" for kw, lv in process_keywords])
            print(f"   → {kw_display}")
    
    # ========== 步骤2: 执行检索 ==========
    # 产品名同义词80条 + 工艺同义词30条 + CAS精确匹配（最多30条）
    print(f"\n📊 开始检索 (产品80条 + 工艺30条 + CAS精确匹配最多30条)...")
    
    # 检测输入是否包含中文
    has_chinese = contains_chinese(product_name)
    
    # 计算产品关键词配额（固定120条，内部按权重分配，多取50%作为候选池）
    product_quota = calculate_retrieval_quota(product_keywords, has_original=not has_chinese, mode="fixed", fixed_quota=120)
    
    # 存储检索结果
    all_docs = []
    all_metadatas = []
    all_distances = []
    all_sources = []  # 记录每个因子的来源关键词
    seen_ids = set()
    cas_matched_ids = set()  # 记录来自CAS精确匹配的因子ID
    
    # CAS号精确匹配（去掉空格和横杠后匹配）
    if cas_number:
        cas_clean = cas_number.replace(" ", "").replace("-", "").strip()
        if cas_clean:
            print(f"   🔍 CAS精确匹配 → 查询中...")
            # 从数据库中精确匹配CAS号
            all_data = collection.get()
            cas_matched_results = []
            for i, meta in enumerate(all_data['metadatas']):
                db_cas = (meta.get('cas_number', '') or '').replace(" ", "").replace("-", "").strip()
                if db_cas == cas_clean:
                    doc_id = meta.get('activity_name', '') + meta.get('product_name', '')
                    cas_matched_results.append({
                        'doc_id': doc_id,
                        'doc': all_data['documents'][i],
                        'meta': meta
                    })
                    if len(cas_matched_results) >= 30:  # 最多30条
                        break
            
            if cas_matched_results:
                print(f"   🔍 CAS精确匹配 → {len(cas_matched_results)}个")
                for item in cas_matched_results:
                    doc_id = item['doc_id']
                    cas_matched_ids.add(doc_id)  # 标记为CAS匹配
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(item['doc'])
                        all_metadatas.append(item['meta'])
                        all_distances.append(0.0)  # 精确匹配，距离为0
                        all_sources.append(f"CAS精确:{cas_number.strip()}")
            else:
                print(f"   🔍 CAS精确匹配 → 0个 (无匹配)")
    
    # 检索产品关键词（按配额分配）
    for keyword, quota in product_quota:
        if keyword is None:
            # 原始输入
            source_name = f"原始输入:{product_name.strip()}"
            print(f"   🔍 {source_name} → {quota}个")
            query_emb = model.encode([product_name.strip()])
            results = collection.query(query_embeddings=query_emb.tolist(), n_results=quota)
        else:
            # 扩展关键词
            source_name = f"同义词:{keyword}"
            print(f"   🔍 {source_name} → {quota}个")
            query_emb = model.encode([keyword])
            results = collection.query(query_embeddings=query_emb.tolist(), n_results=quota)
        
        for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results.get('distances', [[]])[0]):
            doc_id = meta.get('activity_name', '') + meta.get('product_name', '')
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)
                all_metadatas.append(meta)
                all_distances.append(dist)
                all_sources.append(source_name)
    
    # 检索工艺关键词（固定45条，内部按权重分配，多取50%作为候选池）
    if process_keywords:
        process_quota = calculate_retrieval_quota(process_keywords, has_original=not contains_chinese(process), mode="fixed", fixed_quota=45)
        for keyword, quota in process_quota:
            if keyword is None:
                source_name = f"原始工艺:{process.strip()}"
                print(f"   🔍 {source_name} → {quota}个")
                query_emb = model.encode([process.strip()])
                results = collection.query(query_embeddings=query_emb.tolist(), n_results=quota)
            else:
                source_name = f"工艺同义词:{keyword}"
                print(f"   🔍 {source_name} → {quota}个")
                query_emb = model.encode([keyword])
                results = collection.query(query_embeddings=query_emb.tolist(), n_results=quota)
            
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results.get('distances', [[]])[0]):
                doc_id = meta.get('activity_name', '') + meta.get('product_name', '')
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)
                    all_metadatas.append(meta)
                    all_distances.append(dist)
                    all_sources.append(source_name)
    
    # 【关键优化】如果用户未填写工艺，主动补充检索 market for 相关条目
    # 这类条目代表市场平均值，当用户未指定工艺时更合适
    if not process or not process.strip():
        # 为每个产品关键词补充 market for 查询
        for keyword, quota in product_quota:
            if keyword is None:
                search_term = product_name.strip()
            else:
                search_term = keyword
            
            market_query = f"market for {search_term}"
            market_quota = 15  # 每个关键词额外补充15条 market for
            try:
                query_emb = model.encode([market_query])
                results = collection.query(query_embeddings=query_emb.tolist(), n_results=market_quota)
                for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results.get('distances', [[]])[0]):
                    doc_id = meta.get('activity_name', '') + meta.get('product_name', '')
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
                        all_metadatas.append(meta)
                        all_distances.append(dist)
                        all_sources.append(f"Market补充:{search_term}")
            except:
                pass  # 忽略检索失败
    
    # 备注不参与向量检索，只在LLM评分时作为用户输入信息使用
    
    MIN_CANDIDATES = 100
    
    # 记录去重后的初始数量
    initial_count = len(all_docs)
    
    # 如果去重后不足100条，需要额外检索补充
    if initial_count < MIN_CANDIDATES:
        need_more = MIN_CANDIDATES - initial_count
        print(f"   → 候选因子不足{initial_count}条，额外检索{need_more}条...")
        
        # 使用同义词扩展检索，而不是原始产品名
        # 基于已成功的同义词列表，额外增加更多相关能源关键词
        extra_keywords = [
            "energy", "power generation", "energy supply", 
            "electricity generation", "power plant", "energy system",
            "utility", "power system", "electric grid", "energy distribution"
        ]
        
        added = 0
        for keyword in extra_keywords:
            if len(all_docs) >= MIN_CANDIDATES:
                break
            
            extra_results = collection.query(
                query_embeddings=model.encode([keyword]).tolist(),
                n_results=need_more + 20  # 多取一些以便去重
            )
            
            for doc, meta, dist in zip(
                extra_results['documents'][0], 
                extra_results['metadatas'][0], 
                extra_results.get('distances', [[]])[0]
            ):
                if len(all_docs) >= MIN_CANDIDATES:
                    break
                doc_id = meta.get('activity_name', '') + meta.get('product_name', '')
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)
                    all_metadatas.append(meta)
                    all_distances.append(dist)
                    all_sources.append(f"扩展检索:{keyword}")
                    added += 1
        
        print(f"   → 额外检索完成，新增{added}条（去重后共{len(all_docs)}条）")
    
    # 截取TOP_K条作为最终候选
    docs = all_docs[:TOP_K]
    metadatas = all_metadatas[:TOP_K]
    distances = all_distances[:TOP_K]
    sources = all_sources[:TOP_K]
    
    # 传递CAS匹配标记供后续评分使用
    cas_matched_ids_topk = set()
    for meta in metadatas:
        doc_id = meta.get('activity_name', '') + meta.get('product_name', '')
        if doc_id in cas_matched_ids:
            cas_matched_ids_topk.add(doc_id)
    
    # 严格截取为100条
    docs = docs[:MIN_CANDIDATES]
    metadatas = metadatas[:MIN_CANDIDATES]
    distances = distances[:MIN_CANDIDATES]
    sources = sources[:MIN_CANDIDATES]
    
    print(f"   → 去重后候选因子: {len(docs)} 个")
    
    # 显示向量检索结果（每个因子的来源）
    print(f"\n📋 向量检索结果详情:")
    for i, (meta, source) in enumerate(zip(metadatas, sources)):
        product = meta.get('product_name', 'N/A')[:40]
        activity = meta.get('activity_name', 'N/A')[:30]
        print(f"   [{i+1:3d}] {product:<42} | {activity:<32} | 来源:{source}")
    
    if not docs:
        print("❌ 未找到匹配结果")
        return "<div style='padding: 60px; text-align: center; color: #9ca3af;'>未找到匹配结果</div>"
    
    judge_text = " ".join([p for p in [product_name, process, remark] if p])
    
    # 先进行LLM判断（废弃物、设施设备）
    print("\n🤖 AI 预判断中...")
    print(f"   → 废弃物判断: ", end="")
    user_is_waste = judge_waste(judge_text)
    print("是" if user_is_waste == True else ("否" if user_is_waste == False else "不确定"))
    
    print(f"   → 设施/设备判断: ", end="")
    user_is_infra = judge_infrastructure(product_name)
    print("是" if user_is_infra == True else ("否" if user_is_infra == False else "不确定"))
    
    # LLM全量分析候选因子，返回每个的匹配度分数
    print("\n🤖 LLM全量分析候选因子...")
    user_full_input = f"产品: {product_name}\n工艺: {process}\n备注: {remark}"
    
    # 调试：显示输入给LLM的用户输入信息
    print("\n" + "=" * 60)
    print("📋 输入给LLM的用户输入信息:")
    print("=" * 60)
    print(user_full_input)
    print("=" * 60 + "\n")
    
    llm_scores = llm_evaluate_candidates(user_full_input, docs, metadatas)
    
    # 调试：按LLM分数排序显示所有候选因子（代码排序，不是AI）
    print("\n📊 LLM评分排序（全部候选因子，按分数从高到低）:")
    print("-" * 120)
    
    # 构建排序列表：[(产品名, 活动名, 分数, 理由), ...]
    llm_sorted = []
    for i, meta in enumerate(metadatas):
        score_data = llm_scores.get(i, {'score': 50, 'reason': ''})
        llm_sorted.append((
            meta.get('product_name', 'N/A'),
            meta.get('activity_name', 'N/A'),
            score_data['score'],
            score_data['reason']
        ))
    
    # 代码排序：从高到低
    llm_sorted.sort(key=lambda x: -x[2])
    
    # 显示排序结果（包含理由）
    for rank, (prod, act, score, reason) in enumerate(llm_sorted, 1):
        prod_short = prod[:35] if len(prod) > 35 else prod
        act_short = act[:25] if len(act) > 25 else act
        reason_short = reason[:30] if len(reason) > 30 else reason
        print(f"{rank:>3}. {prod_short:<37} | {act_short:<27} | {score:>3}分 | {reason_short}")
    
    print("-" * 120)
    
    # 在LLM分数基础上应用规则扣分，并显示所有因子的分数
    print("\n📈 LLM匹配结果 (全部候选因子):")
    print("-" * 100)
    print(f"{'序号':>4} {'产品名称':<45} {'LLM分':>6} {'规则扣分':>8} {'最终分':>6}")
    print("-" * 100)
    
    scored_results = []
    # 判断用户是否填写了生产工艺（规则2的触发条件）
    user_has_process = bool(process and process.strip())
    
    for i, (doc, meta) in enumerate(zip(docs, metadatas)):
        cut_off = meta.get('cut_off', '')
        candidate_sector = meta.get('sector', '')
        candidate_is_infra = "infrastructure" in candidate_sector.lower() and "machinery" in candidate_sector.lower()
        
        # 判断是否为CAS精确匹配的因子
        doc_id = meta.get('activity_name', '') + meta.get('product_name', '')
        is_cas_matched = doc_id in cas_matched_ids_topk
        
        # 获取候选因子的产品名和活动名，判断是否为因子A（低信息量activity）
        candidate_product = meta.get('product_name', '')
        candidate_activity = meta.get('activity_name', '')
        is_factor_a = is_info_less_activity(candidate_product, candidate_activity)
        
        # LLM基础分数和理由
        score_data = llm_scores.get(i, {'score': 50, 'reason': ''})
        llm_base_score = score_data['score']
        llm_reason = score_data['reason']
        
        # CAS精确匹配的因子：LLM初始分至少85分
        if is_cas_matched and llm_base_score < 85:
            llm_base_score = 85
            llm_reason = "CAS精确匹配保底85分"
        
        # 在LLM分数基础上应用规则扣分
        # activity被置空（即is_factor_a为True）
        final_score, deductions = calculate_final_score(
            llm_base_score, 
            user_is_waste, 
            cut_off, 
            user_is_infra, 
            candidate_is_infra,
            is_factor_a,
            user_has_process
        )
        
        # 计算总扣分
        total_deduct = sum(d['扣分'] for d in deductions)
        
        scored_results.append({
            'meta': meta,
            'score': final_score,
            'llm_score': llm_base_score,  # 保存LLM原始分数（已应用CAS保底）
            'llm_reason': llm_reason,     # 保存LLM评分理由
            'deductions': deductions,
            'is_waste': cut_off.lower() in ["waste", "recyclable"],
            'is_cas_matched': is_cas_matched,  # 是否来自CAS精确匹配
            'geographies': generate_random_geographies(),
            'time_period': random.choice(TIME_RANGES)
        })
        
        # 显示每行
        product_name_short = meta.get('product_name', 'N/A')[:43]
        deduct_str = str(total_deduct) if total_deduct != 0 else "-"
        print(f"{i+1:>4} {product_name_short:<45} {llm_base_score:>6} {deduct_str:>8} {final_score:>6}")
    
    print("-" * 100)
    
    # 如果用户未填写工艺，对 market for / market group for 类活动加分（适度 +3 分）
    # 这类活动代表市场平均值，当用户未指定工艺时更合适
    if not process or not process.strip():
        market_bonus = 3
        market_count = 0
        for r in scored_results:
            activity = r['meta'].get('activity_name', '').lower()
            if activity.startswith('market for ') or activity.startswith('market group for '):
                r['score'] += market_bonus
                market_count += 1
        if market_count > 0:
            print(f"   → Market因子加权: +{market_bonus}分, 共 {market_count} 个因子受益")
    
    scored_results.sort(key=lambda x: (-x['score'], x.get('distance', 0)))
    
    # 筛选>=60分的，至少6条，最多30条
    filtered_results = [r for r in scored_results if r['score'] >= 60]
    if len(filtered_results) < MIN_RESULTS:
        # 如果少于6条，从全部结果中取分数最高的补充到6条
        top_results = scored_results[:MIN_RESULTS]
    else:
        top_results = filtered_results[:MAX_RESULTS]
    
    print(f"\n✅ 检索完成! 找到 {len(top_results)} 个推荐因子 (共 {len(filtered_results)} 条≥60分)")
    perfect = sum(1 for r in scored_results if r['score'] >= 95)
    high_score = sum(1 for r in scored_results if r['score'] >= 80)
    print(f"   → 优秀匹配(≥95分): {perfect} 个 | 良好匹配(≥80分): {high_score} 个")
    
    # 格式化关键词用于显示
    if product_keywords:
        product_kw_str = ", ".join([f"{kw}({lv})" for kw, lv in product_keywords[:5]])
    else:
        product_kw_str = "无"
    
    if process_keywords:
        process_kw_str = ", ".join([f"{kw}({lv})" for kw, lv in process_keywords])
    else:
        process_kw_str = "无"
    
    # 现代卡片风格CSS
    css = """
    <style>
    * { box-sizing: border-box; }
    body { margin: 0 !important; padding: 0 !important; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important; background: #f8fafc !important; }
    
    .result-wrapper {
        max-width: 1400px !important;
        margin: 0 auto !important;
        padding: 24px !important;
        background: #f8fafc !important;
    }
    
    .result-wrapper * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* 顶部统计 */
    .stats-bar {
        display: flex !important;
        gap: 16px !important;
        margin-bottom: 24px !important;
    }
    .stat-card {
        flex: 1 !important;
        background: white !important;
        border-radius: 12px !important;
        padding: 20px 24px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
        border-left: 4px solid #22c55e !important;
    }
    .stat-card.warning { border-left-color: #f59e0b !important; }
    .stat-card.info { border-left-color: #3b82f6 !important; }
    .stat-num {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        line-height: 1 !important;
    }
    .stat-label {
        font-size: 13px !important;
        color: #64748b !important;
        margin-top: 6px !important;
    }
    
    /* 筛选器栏 */
    .filter-bar {
        background: white !important;
        border-radius: 12px !important;
        padding: 16px 24px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
        display: flex !important;
        gap: 32px !important;
        flex-wrap: wrap !important;
        align-items: center !important;
    }
    .filter-item {
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
    }
    .filter-label {
        font-size: 14px !important;
        color: #64748b !important;
    }
    .filter-badge {
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    .badge-green { color: #16a34a !important; }
    .badge-red { color: #dc2626 !important; }
    .badge-gray { color: #94a3b8 !important; }
    
    /* 关键词 */
    .keywords-section {
        background: white !important;
        border-radius: 12px !important;
        padding: 16px 24px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }
    .section-title {
        font-size: 12px !important;
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 12px !important;
        font-weight: 600 !important;
    }
    .keyword-tags {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
    }
    .kw-tag {
        padding: 6px 12px !important;
        background: #f0fdf4 !important;
        border: 1px solid #bbf7d0 !important;
        border-radius: 6px !important;
        font-size: 13px !important;
        color: #166534 !important;
    }
    .kw-tag.process { background: #eff6ff !important; border-color: #bfdbfe !important; color: #1e40af !important; }
    
    /* 结果卡片 */
    .result-list {
        display: flex !important;
        flex-direction: column !important;
        gap: 16px !important;
    }
    .result-card {
        background: white !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
        border-left: 5px solid #22c55e !important;
        margin-bottom: 0 !important;
    }
    .result-card:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.12) !important;
    }
    
    /* 卡片主体 */
    .card-main {
        display: grid !important;
        grid-template-columns: 1fr auto !important;
        gap: 32px !important;
        padding: 28px 32px !important;
        align-items: start !important;
    }
    
    /* 左侧 */
    .card-left { flex: 1 !important; }
    .card-header-row {
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        margin-bottom: 16px !important;
    }
    .rank-badge {
        background: #22c55e !important;
        color: white !important;
        padding: 4px 10px !important;
        border-radius: 6px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }
    .type-tag {
        padding: 4px 10px !important;
        border-radius: 6px !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
    }
    .type-product { background: #dbeafe !important; color: #1e40af !important; }
    .type-waste { background: #fef3c7 !important; color: #b45309 !important; }
    
    /* 名称 */
    .product-name-container {
        display: flex !important;
        align-items: baseline !important;
        gap: 10px !important;
        margin-bottom: 10px !important;
    }
    .name-tag {
        padding: 3px 8px !important;
        border-radius: 4px !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        flex-shrink: 0 !important;
    }
    .name-tag.product { background: #22c55e !important; color: white !important; }
    .name-tag.activity { background: #8b5cf6 !important; color: white !important; }
    
    .product-name {
        font-size: 17px !important;
        font-weight: 600 !important;
        color: #0f172a !important;
        margin: 0 !important;
        line-height: 1.4 !important;
    }
    .activity-name {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #475569 !important;
        margin: 0 0 20px 0 !important;
        line-height: 1.5 !important;
    }
    
    /* 理由tooltip */
    .llm-reason {
        color: #b45309 !important;
        cursor: pointer !important;
        border-bottom: 1px dashed #b45309 !important;
        max-width: 300px !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        position: relative !important;
    }
    .llm-reason:hover::after {
        content: attr(data-full) !important;
        position: absolute !important;
        top: -8px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        background: #1f2937 !important;
        color: white !important;
        padding: 6px 10px !important;
        border-radius: 6px !important;
        font-size: 12px !important;
        white-space: pre-wrap !important;
        max-width: 300px !important;
        z-index: 1000 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
    }
    
    /* 信息网格 */
    .info-grid {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 24px !important;
    }
    .info-item { min-width: 140px !important; }
    .info-label {
        font-size: 11px !important;
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 4px !important;
    }
    .info-value {
        font-size: 13px !important;
        color: #334155 !important;
        font-weight: 500 !important;
    }
    .sector-tag {
        display: inline-block !important;
        background: #f1f5f9 !important;
        color: #64748b !important;
        padding: 4px 10px !important;
        border-radius: 6px !important;
        font-size: 12px !important;
        margin-top: 12px !important;
    }
    
    /* 卡片头部 - 包含排名+评分区 */
    .card-header-compact {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        padding: 16px 24px !important;
        background: white !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }
    .card-header-left {
        display: flex !important;
        align-items: center !important;
        gap: 16px !important;
    }
    .card-header-right {
        display: flex !important;
        align-items: center !important;
        gap: 16px !important;
    }
    
    /* 紧凑分数 */
    .score-compact {
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    .score-circle-compact {
        width: 48px !important;
        height: 48px !important;
        border-radius: 50% !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .score-circle-compact.high { background: #dcfce7 !important; border: 2px solid #22c55e !important; }
    .score-circle-compact.medium { background: #fef3c7 !important; border: 2px solid #f59e0b !important; }
    .score-circle-compact.low { background: #fee2e2 !important; border: 2px solid #ef4444 !important; }
    .score-num-compact { font-size: 18px !important; font-weight: 700 !important; line-height: 1 !important; }
    .score-circle-compact.high .score-num-compact { color: #16a34a !important; }
    .score-circle-compact.medium .score-num-compact { color: #d97706 !important; }
    .score-circle-compact.low .score-num-compact { color: #dc2626 !important; }
    
    /* LLM评分标签 */
    .llm-badge {
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        padding: 6px 12px !important;
        background: #fffbeb !important;
        border: 1px solid #fef3c7 !important;
        border-radius: 6px !important;
        font-size: 12px !important;
    }
    .llm-label { color: #92400e !important; }
    .llm-score { font-weight: 600 !important; color: #a16207 !important; }
    .llm-reason { color: #b45309 !important; max-width: 150px !important; overflow: hidden !important; text-overflow: ellipsis !important; white-space: nowrap !important; }
    .deduct-badge {
        padding: 4px 8px !important;
        border-radius: 4px !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        background: #fee2e2 !important;
        color: #dc2626 !important;
    }
    
    /* 地理位置 */
    .geo-section {
        padding: 12px 24px !important;
        border-top: 1px solid #f1f5f9 !important;
        background: #fafafa !important;
    }
    .geo-title {
        font-size: 11px !important;
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 8px !important;
        font-weight: 600 !important;
    }
    .geo-list { display: flex !important; flex-wrap: wrap !important; gap: 6px !important; }
    .geo-item {
        padding: 3px 8px !important;
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 4px !important;
        font-size: 11px !important;
        color: #475569 !important;
    }
    
    /* 分数区 */
    .card-score {
        text-align: center !important;
        min-width: 80px !important;
    }
    .score-circle {
        width: 64px !important;
        height: 64px !important;
        border-radius: 50% !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 auto 8px !important;
    }
    .score-circle.high { background: #dcfce7 !important; border: 3px solid #22c55e !important; }
    .score-circle.medium { background: #fef3c7 !important; border: 3px solid #f59e0b !important; }
    .score-circle.low { background: #fee2e2 !important; border: 3px solid #ef4444 !important; }
    .score-num { font-size: 22px !important; font-weight: 700 !important; line-height: 1 !important; }
    .score-circle.high .score-num { color: #16a34a !important; }
    .score-circle.medium .score-num { color: #d97706 !important; }
    .score-circle.low .score-num { color: #dc2626 !important; }
    .score-label { font-size: 11px !important; color: #64748b !important; }
    .status-tag {
        font-size: 12px !important;
        font-weight: 600 !important;
    }
    .status-excellent { color: #16a34a !important; }
    .status-good { color: #2563eb !important; }
    .status-normal { color: #d97706 !important; }
    .status-review { color: #dc2626 !important; }
    
    /* 分页控件 */
    .pagination-container {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 8px !important;
        margin-top: 24px !important;
        padding: 16px 0 !important;
    }
    .pagination-btn {
        padding: 8px 14px !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        background: white !important;
        color: #334155 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        min-width: 40px !important;
    }
    .pagination-btn:hover:not(:disabled) {
        background: #f0fdf4 !important;
        border-color: #22c55e !important;
        color: #16a34a !important;
    }
    .pagination-btn:disabled {
        opacity: 0.5 !important;
        cursor: not-allowed !important;
    }
    .pagination-btn.active {
        background: #22c55e !important;
        border-color: #22c55e !important;
        color: white !important;
    }
    .pagination-info {
        font-size: 13px !important;
        color: #64748b !important;
        margin: 0 12px !important;
    }
    .page-ellipsis {
        padding: 8px 4px !important;
        color: #94a3b8 !important;
    }

    </style>
    """
    
    html = ['<div class="result-wrapper">', css]
    
    # 顶部统计
    html.append(f"""
    <div class="stats-bar">
        <div class="stat-card">
            <div class="stat-num">{len(top_results)}</div>
            <div class="stat-label">匹配因子</div>
        </div>
        <div class="stat-card warning">
            <div class="stat-num">{sum(1 for r in top_results if r['score'] >= 90)}</div>
            <div class="stat-label">高分推荐</div>
        </div>
        <div class="stat-card info">
            <div class="stat-num">{sum(1 for r in top_results if not r['deductions'])}</div>
            <div class="stat-label">完全匹配</div>
        </div>
    </div>
    """)
    
    # LLM判断 + 关键词
    waste_str = "是" if user_is_waste == True else ("否" if user_is_waste == False else "不确定")
    infra_str = "是" if user_is_infra == True else ("否" if user_is_infra == False else "不确定")
    
    html.append(f"""
    <div class="filter-bar">
        <div class="filter-item">
            <span class="filter-label">废弃物判断</span>
            <span class="filter-badge {'badge-red' if user_is_waste == True else ('badge-green' if user_is_waste == False else 'badge-gray')}">
                {'✓ ' if user_is_waste == True else ('✗ ' if user_is_waste == False else '')}{waste_str}
            </span>
        </div>
        <div class="filter-item">
            <span class="filter-label">设施/设备判断</span>
            <span class="filter-badge {'badge-red' if user_is_infra == True else ('badge-green' if user_is_infra == False else 'badge-gray')}">
                {'✓ ' if user_is_infra == True else ('✗ ' if user_is_infra == False else '')}{infra_str}
            </span>
        </div>
    </div>
    """)
    
    # 关键词
    html.append(f"""
    <div class="keywords-section">
        <div class="section-title">检索关键词</div>
        <div class="keyword-tags">
            <span class="kw-tag">产品: {product_kw_str if product_kw_str != '无' else '-'}</span>
            <span class="kw-tag process">工艺: {process_kw_str if process_kw_str != '无' else '-'}</span>
        </div>
    </div>
    """)
    
    # 生成卡片（新版列表样式）
    html.append('<div class="result-list" id="resultWrapper">')
    
    # 计算分页参数
    total_results = len(top_results)
    items_per_page = 10
    total_pages = max(1, (total_results + items_per_page - 1) // items_per_page)
    
    for i, item in enumerate(top_results, 1):
        meta = item['meta']
        score = item['score']
        llm_score = item.get('llm_score', score)
        llm_reason = item.get('llm_reason', '')
        deductions = item['deductions']
        is_waste = item['is_waste']
        geos = item['geographies']
        time_period = item['time_period']
        
        # 分数圆圈样式
        if score >= 90:
            score_circle_class = "high"
            status_class, status_text = "status-excellent", "优秀"
        elif score >= 80:
            score_circle_class = "high"
            status_class, status_text = "status-good", "良好"
        elif score >= 60:
            score_circle_class = "medium"
            status_class, status_text = "status-normal", "一般"
        else:
            score_circle_class = "low"
            status_class, status_text = "status-review", "需审核"
        
        # 提取扣分
        rule1_deduct = 0
        rule3_deduct = 0
        for d in deductions:
            if "规则1" in d["规则"]:
                rule1_deduct = d["扣分"]
            elif "规则3" in d["规则"]:
                rule3_deduct = d["扣分"]
        
        product_name_display = meta.get('product_name', 'N/A')
        activity_name = meta.get('activity_name', 'N/A')
        sector = meta.get('sector', 'N/A')
        cut_off = meta.get('cut_off', 'N/A')
        special_activity = meta.get('special_activity', 'N/A')
        cas_number_db = meta.get('cas_number', '')
        
        # 地理位置（统一样式）
        geo_items = ""
        for geo in geos:
            geo_items += f'<span class="geo-item">{geo}</span>'
        
        # 构建基本信息区的底部内容（包含其他字段）
        basic_info_items = f"""
                        <div class="info-item">
                            <div class="info-label">Special Activity Type</div>
                            <div class="info-value">{special_activity}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Time Period</div>
                            <div class="info-value">{time_period}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Cut-Off Classification</div>
                            <div class="info-value">{cut_off}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">CAS Number</div>
                            <div class="info-value">{cas_number_db if cas_number_db else '-'}</div>
                        </div>
        """
        
        # 扣分显示
        deduct_badge = ""
        if rule1_deduct:
            deduct_badge += f'<span class="deduct-badge">规则1 -{rule1_deduct}</span>'
        if rule3_deduct:
            deduct_badge += f'<span class="deduct-badge">规则3 -{rule3_deduct}</span>'
        if not deductions:
            deduct_badge = '<span class="deduct-badge" style="background:#dcfce7;color:#16a34a;">无扣分</span>'
        
        # 计算当前因子所属页码
        current_page = (i - 1) // items_per_page + 1
        page_display_class = "page-1" if current_page == 1 else f"page-{current_page}"
        page_display_style = "" if current_page == 1 else "display:none;"
        
        html.append(f"""
        <div class="result-card {page_display_class}" data-page="{current_page}" style="{page_display_style}">
            <div class="card-header-compact">
                <div class="card-header-left">
                    <span class="rank-badge">#{i}</span>
                    <div class="llm-badge">
                        <span class="llm-label">LLM评分:</span>
                        <span class="llm-score">{llm_score}分</span>
                        {f'<span class="llm-reason" data-full="理由:{llm_reason}">理由:{llm_reason[:50]}</span>' if llm_reason else ''}
                        {deduct_badge}
                    </div>
                </div>
                <div class="card-header-right">
                    <div class="score-compact">
                        <div class="score-circle-compact {score_circle_class}">
                            <span class="score-num-compact">{score}</span>
                        </div>
                        <span class="status-tag {status_class}">{status_text}</span>
                    </div>
                </div>
            </div>
            <div class="card-main">
                <div class="card-left">
                    <div class="product-name-container">
                        <span class="name-tag product">PRODUCT</span>
                        <h3 class="product-name">{product_name_display}</h3>
                    </div>
                    <div class="product-name-container">
                        <span class="name-tag activity">ACTIVITY</span>
                        <p class="activity-name">{activity_name}</p>
                    </div>
                    <div class="info-grid">
                        {basic_info_items}
                    </div>
                    <span class="sector-tag">{sector}</span>
                </div>
            </div>
            <div class="geo-section">
                <div class="geo-title">Geography</div>
                <div class="geo-list">{geo_items}</div>
            </div>
        </div>
        """)
    
    html.append('</div>')
    
    # 添加分页控件
    pagination_html = f'''
    <div class="pagination-container" id="paginationContainer" style="margin-top:24px;padding:16px 0;">
        <button class="pagination-btn" id="prevBtn" onclick="window.goToFactorPage(currentPage - 1)">‹ 上一页</button>
    '''
    
    # 生成页码按钮
    page_buttons = ""
    for p in range(1, total_pages + 1):
        if p == 1:
            page_buttons += f'<button class="pagination-btn active" onclick="window.goToFactorPage({p})">{p}</button>'
        elif p <= 3 or p > total_pages - 2 or abs(p - 1) <= 1:
            page_buttons += f'<button class="pagination-btn" onclick="window.goToFactorPage({p})">{p}</button>'
        elif p == 4 and total_pages > 5:
            page_buttons += '<span class="page-ellipsis">...</span>'
        elif p == total_pages - 3 and total_pages > 5:
            page_buttons += '<span class="page-ellipsis">...</span>'
    
    pagination_html += page_buttons
    pagination_html += f'''
        <button class="pagination-btn" id="nextBtn" onclick="window.goToFactorPage(currentPage + 1)">下一页 ›</button>
        <span class="pagination-info" id="pageInfo">共 {total_results} 条，第 1/{total_pages} 页</span>
    </div>
    
    <script>
    (function() {{
        window.currentPage = 1;
        window.totalPages = {total_pages};
        
        window.goToFactorPage = function(page) {{
            if (page < 1 || page > window.totalPages || page === window.currentPage) return;
            
            // 隐藏所有卡片
            var allCards = document.querySelectorAll('div[data-page]');
            for (var i = 0; i < allCards.length; i++) {{
                allCards[i].style.display = "none";
            }}
            
            // 显示目标页卡片
            var targetCards = document.querySelectorAll('div[data-page="' + page + '"]');
            for (var j = 0; j < targetCards.length; j++) {{
                targetCards[j].style.display = "";
            }}
            
            // 更新页码按钮状态
            var btns = document.querySelectorAll('.pagination-btn');
            for (var k = 0; k < btns.length; k++) {{
                var btn = btns[k];
                if (btn.id === 'prevBtn' || btn.id === 'nextBtn') continue;
                if (btn.textContent.trim() == page) {{
                    btn.classList.add('active');
                }} else {{
                    btn.classList.remove('active');
                }}
            }}
            
            // 更新按钮状态
            var prevBtn = document.getElementById('prevBtn');
            var nextBtn = document.getElementById('nextBtn');
            if (prevBtn) prevBtn.disabled = (page === 1);
            if (nextBtn) nextBtn.disabled = (page === window.totalPages);
            
            // 更新页码信息
            var info = document.getElementById('pageInfo');
            if (info) {{
                info.textContent = "共 {total_results} 条，第 " + page + "/" + window.totalPages + " 页";
            }}
            
            window.currentPage = page;
            
            // 滚动到结果区域顶部
            var wrapper = document.getElementById('resultWrapper');
            if (wrapper) {{
                wrapper.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            }}
        }};
    }})();
    </script>
    '''
    
    html.append(pagination_html)
    html.append('</div>')  # result-wrapper
    return '\n'.join(html)

def main():
    init_models()
    
    # 创建Blocks实例 - 绿色主题
    demo = gr.Blocks(title="AI 碳足迹因子选择器")
    
    # CSS样式 - Gradio 6.0 兼容
    css = """
    /* 全局重置 */
    * { box-sizing: border-box; margin: 0; padding: 0; }
    .gradio-container { max-width: 100% !important; width: 100% !important; min-height: 100vh !important; }
    .gradio-header, .gradio-footer { display: none !important; }
    
    /* 左右布局容器 */
    .main-row { 
        display: flex !important; 
        flex-direction: row !important; 
        gap: 24px !important; 
        padding: 24px !important; 
        background: #f8fafc !important; 
        height: 100vh !important; 
        overflow: hidden !important;
    }
    
    /* 左侧面板 - 绿色主题卡片 */
    .left-panel {
        width: 400px !important;
        min-width: 380px !important;
        background: white !important;
        border-radius: 16px !important;
        border: 2px solid #22c55e !important;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.15) !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
    }
    
    /* 左侧面板标题区 */
    .left-header {
        padding: 20px 24px !important;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%) !important;
        border-bottom: 1px solid #bbf7d0 !important;
    }
    .left-header h1 {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #15803d !important;
        margin: 0 0 4px 0 !important;
    }
    .left-header p {
        font-size: 12px !important;
        color: #16a34a !important;
        margin: 0 !important;
    }
    
    /* 查询条件标签 */
    .query-label {
        padding: 12px 24px !important;
        border-bottom: 1px solid #d1fae5 !important;
    }
    .query-label span {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #166534 !important;
    }
    
    /* 表单区域 */
    .left-panel > div {
        padding: 16px 24px !important;
        flex: 1 !important;
        overflow-y: auto !important;
    }
    
    /* 输入框样式 - Gradio 6.0 */
    .left-panel .label-wrap {
        display: block !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #374151 !important;
        margin-bottom: 6px !important;
    }
    .left-panel textarea {
        margin-bottom: 12px !important;
    }
    .left-panel input, 
    .left-panel textarea {
        width: 100% !important;
        padding: 10px 12px !important;
        font-size: 14px !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        background: #f9fafb !important;
        color: #1f2937 !important;
    }
    .left-panel input:focus, 
    .left-panel textarea:focus {
        border-color: #22c55e !important;
        box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.2) !important;
        outline: none !important;
    }
    
    /* 手动表单标签样式 */
    .left-panel :is(p, h1, h2, h3, h4, h5, h6) {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #374151 !important;
        margin-bottom: 6px !important;
        margin-top: 8px !important;
    }
    .left-panel :is(p, h1, h2, h3, h4, h5, h6):first-child {
        margin-top: 0 !important;
    }
    .left-panel :is(p, h1, h2, h3, h4, h5, h6) em {
        font-style: normal !important;
        color: #dc2626 !important;
    }
    
    /* 搜索按钮 */
    .search-btn button {
        width: calc(100% - 48px) !important;
        margin: 0 24px 12px !important;
        padding: 12px 20px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        cursor: pointer !important;
    }
    .search-btn button:hover {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
    }
    
    /* 提示区 */
    .hint-area {
        padding: 12px 24px 16px !important;
        background: #f0fdf4 !important;
    }
    .hint-area p {
        font-size: 12px !important;
        color: #15803d !important;
        line-height: 1.5 !important;
        padding: 10px 14px !important;
        background: white !important;
        border: 1px solid #86efac !important;
        border-radius: 6px !important;
    }
    
    /* 右侧结果区 */
    .right-panel {
        flex: 1 !important;
        background: white !important;
        border-radius: 16px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
        overflow: hidden !important;
    }
    
    /* 滚动条 */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-thumb { background: #22c55e; border-radius: 3px; }
    """
    
    with demo:
        with gr.Row(elem_classes=["main-row"]):
            # 左侧输入面板
            with gr.Column(elem_classes=["left-panel"]):
                gr.HTML('''
                    <div class="left-header">
                        <h1>Carbon Footprint Factor Selector</h1>
                        <p>智能碳足迹因子选择器</p>
                    </div>
                    <div class="query-label">
                        <span>查询条件</span>
                    </div>
                ''', elem_classes=["left-header"])
                
                gr.Markdown("**CAS号**")
                cas_input = gr.Textbox(placeholder="如：7439-89-6（可选）", lines=1)
                gr.Markdown("**名称** *（必填）*")
                product_input = gr.Textbox(placeholder="如：生铁、碳素钢", lines=1)
                gr.Markdown("**工艺/路线**")
                process_input = gr.Textbox(placeholder="如：转炉炼钢（可选）", lines=1)
                gr.Markdown("**备注**")
                remark_input = gr.Textbox(placeholder="用途说明（可选）", lines=2)
                
                gr.HTML('<div class="hint-area"><p>💡 填写产品名称后点击 Search，系统返回评分≥60分的候选因子（最少6条，最多30条）</p></div>')
            
            # 右侧结果显示
            with gr.Column(elem_classes=["right-panel"]):
                output = gr.HTML("<div style='color:#9ca3af;text-align:center;padding:60px;'>输入产品名称开始检索</div>")
        
        # 搜索按钮 - 使用CSS定位
        gr.Button("🔍 Search", elem_classes=["search-btn"], size="lg").click(
            fn=search_and_recommend, 
            inputs=[cas_input, product_input, process_input, remark_input], 
            outputs=output
        )
    
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Default(primary_hue="green", secondary_hue="emerald"), css=css)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"""
========================================
程序崩溃时间: {datetime.datetime.now()}
错误类型: {type(e).__name__}
错误信息: {str(e)}

完整错误追踪:
{traceback.format_exc()}
========================================
"""
        print(error_msg)
        # 保存到日志文件
        with open("error.log", "a", encoding="utf-8") as f:
            f.write(error_msg)
        input("\n按回车键退出...")
