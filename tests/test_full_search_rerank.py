import requests
import json

def test_complete_flow():
    """
    测试完整流程：
    1. 调用 /search 获取 60 条结果
    2. 将搜索结果的 body（JSON 字符串）传给 /rerank 进行重排序
    """
    api_base_url = "http://localhost:8000"
    
    # ==========================================
    # 第一步：测试搜索接口 /search
    # ==========================================
    search_url = f"{api_base_url}/search"
    search_payload = {
        "query": "DPF_OUTP_HIGH_ERROR有哪些标定参数？",
        "top_k": 60,
        "strategy": "global"
    }
    
    print("--- [Step 1] 测试搜索接口 /search ---")
    try:
        search_response = requests.post(search_url, json=search_payload)
        search_response.raise_for_status()
        search_data = search_response.json()
        
        # 之前误以为有嵌套的 'body'，现在还原为直接的 SearchResponse 格式
        search_body_str = json.dumps(search_data)
        
        print(f"✅ 搜索成功！获取到 {len(search_data.get('pure_documents', []))} 条结果。")
        print(f"预览前 2 条: {search_data.get('pure_documents', [])[:2]}")
        
    except Exception as e:
        print(f"❌ 搜索接口调用失败: {e}")
        return

    # ==========================================
    # 第二步：测试重排序接口 /rerank
    # ==========================================
    rerank_url = f"{api_base_url}/rerank"
    rerank_payload = {
        "model": "bge-reranker-v2-m3",
        "query": "DPF_OUTP_HIGH_ERROR有哪些标定参数？",
        "documents": search_body_str,  # 直接传入 Step 1 返回的 JSON 字符串
        "top_k": 3,
        "score_threshold": -5.0
    }
    
    print("\n--- [Step 2] 测试重排序接口 /rerank ---")
    try:
        rerank_response = requests.post(rerank_url, json=rerank_payload)
        rerank_response.raise_for_status()
        rerank_data = rerank_response.json()
        
        sorted_docs = rerank_data.get("pure_documents", [])
        formatted_result = rerank_data.get("formatted_result", "")
        
        print(f"✅ 重排序成功！返回 {len(sorted_docs)} 条满足阈值且截断后的结果。")
        
        print("\n📝 最终格式化输出结果:")
        print("-" * 30)
        print(formatted_result)
        print("-" * 30)
            
    except Exception as e:
        print(f"❌ 重排序接口调用失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"错误详情: {e.response.text}")

if __name__ == "__main__":
    print("🚀 开始 API 联合测试...\n")
    test_complete_flow()
