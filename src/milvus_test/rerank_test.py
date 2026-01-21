"""
测试使用 OpenAI 格式调用 Xinference Rerank 模型
使用 HTTP 直接调用 Xinference 的 v1 API
"""

import requests


def test_openai_format_rerank():
    """测试使用 HTTP 调用 Xinference rerank 模型"""
    try:
        base_url = "http://localhost:9997/v1"
        

        print("📡 正在连接到 Xinference API...")

        # 列出可用模型
        print("\n🔍 正在列出可用模型...")
        response = requests.get(f"{base_url}/models")
        response.raise_for_status()
        models = response.json()

        if models.get("data"):
            print(f"✅ 找到 {len(models['data'])} 个模型:")
            for model in models["data"]:
                print(f"  - {model['id']}")

        # 查找 rerank 模型
        rerank_model_id = None
        for model in models.get("data", []):
            if "rerank" in model["id"].lower():
                rerank_model_id = model["id"]
                break

        if not rerank_model_id:
            print("\n❌ 未找到 rerank 模型，请先启动模型")
            return False

        print(f"\n🎯 使用 rerank 模型: {rerank_model_id}")

        # 测试 rerank
        print("\n🧪 正在测试 rerank 功能...")
        query = "A man is eating pasta."
        documents = [
            "A man is eating food.",
            "A man is eating a piece of bread.",
            "The girl is carrying a baby.",
            "A man is riding a horse.",
            "A woman is playing violin.",
        ]

        # 调用 rerank API
        rerank_url = f"{base_url}/rerank"
        payload = {"model": rerank_model_id, "query": query, "documents": documents}

        response = requests.post(rerank_url, json=payload)
        response.raise_for_status()
        result = response.json()

        print("\n📊 Rerank 结果:")
        if "results" in result:
            for i, item in enumerate(result["results"], 1):
                doc = documents[item["index"]]
                print(f"  {i}. [{item['index']}] {doc}")
                print(f"     相关性分数: {item['relevance_score']:.4f}\n")
        else:
            print(f"响应内容: {result}")

        print("✅ 测试完成！")
        return True

    except Exception as e:
        print(f"❌ 错误: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_openai_format_rerank()
