"""API并发性能测试 - 使用项目配置"""
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志（静默模式）
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 从app_streamlit复制配置
from openai import OpenAI

DEEPSEEK_API_KEYS = [
    "sk-047bd10a8be74a42a60f881fe3a55e9a",
    "sk-3dfefa654b4a48f991aeda80e610f410",
    "sk-91e55627ff874a5ca95dd5990110b187",
    "sk-2dada24f757c413aac965ba6215a8016",
]
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_api_key_index = 0
import threading
_api_lock = threading.Lock()

def get_next_client():
    global _api_key_index
    with _api_lock:
        api_key = DEEPSEEK_API_KEYS[_api_key_index % len(DEEPSEEK_API_KEYS)]
        _api_key_index += 1
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

def call_deepseek(messages, temperature=0.3):
    client = get_next_client()
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature
    )
    return response

def test_single_call(user_input, candidates, call_id):
    """单次API调用"""
    context_lines = []
    for j, c in enumerate(candidates):
        context_lines.append(f"【候选{j+1}】\n产品名称: {c['product_name']}\n活动名称: {c['activity_display']}")
    context = "\n\n".join(context_lines)
    total_candidates = len(candidates)
    
    prompt = f"""用户需求：{user_input}

候选因子清单（共{total_candidates}个）：
{context}

请对上述每个候选因子独立打分（0-100分），输出格式：
1: 85 理由：xxx
2: 72 理由：xxx
...
"""
    
    start = time.time()
    try:
        response = call_deepseek([{"role": "user", "content": prompt}])
        elapsed = time.time() - start
        return {"call_id": call_id, "elapsed": elapsed, "success": True}
    except Exception as e:
        elapsed = time.time() - start
        return {"call_id": call_id, "elapsed": elapsed, "success": False, "error": str(e)[:100]}

def run_performance_test():
    """性能测试"""
    print("=" * 60)
    print("API并发性能测试")
    print("=" * 60)
    
    # 测试数据
    user_input = "产品: 酯类\n工艺: 生产酯类化合物\n备注: 用于工业生产"
    candidates = [
        {"product_name": f"酯类产品{i}", "activity_display": f"酯类生产{i}"}
        for i in range(5)  # 测试5个候选因子
    ]
    
    # 测试1：单次API调用
    print("\n[测试1] 单次API调用")
    result = test_single_call(user_input, candidates, 1)
    if result["success"]:
        print(f"   单次调用耗时: {result['elapsed']:.2f}秒")
        single_avg = result['elapsed']
    else:
        print(f"   调用失败: {result.get('error')}")
        return
    
    # 测试2：5次并行调用
    print("\n[测试2] 5次并行API调用")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(test_single_call, user_input, candidates, i) for i in range(5)]
        
        start_total = time.time()
        results = []
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            status = "OK" if r["success"] else "FAIL"
            print(f"   请求{r['call_id']+1}完成 [{status}]: {r['elapsed']:.2f}秒")
        total_elapsed = time.time() - start_total
    
    success_count = sum(1 for r in results if r["success"])
    avg_elapsed = sum(r["elapsed"] for r in results) / len(results)
    max_elapsed = max(r["elapsed"] for r in results)
    
    print(f"\n[结果汇总]")
    print(f"   成功数: {success_count}/5")
    print(f"   平均单次耗时: {avg_elapsed:.2f}秒")
    print(f"   最长单次耗时: {max_elapsed:.2f}秒")
    print(f"   并行总耗时: {total_elapsed:.2f}秒")
    
    # 估算100个候选因子的耗时
    print(f"\n[100个候选因子耗时估算]")
    print(f"   串行(单个): 100 x {single_avg:.2f} = {100 * single_avg:.2f}秒")
    print(f"   并行(5并发): 100次/5并发 x {avg_elapsed:.2f} = {20 * avg_elapsed:.2f}秒")
    print(f"   并行(10并发): 100次/10并发 x {max_elapsed:.2f} = {10 * max_elapsed:.2f}秒")
    
    # 并行效率
    speedup = (100 * single_avg) / total_elapsed
    print(f"\n[并行加速比]: {speedup:.1f}x")

if __name__ == "__main__":
    run_performance_test()
