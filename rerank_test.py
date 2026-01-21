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
        
        # 补全更多测试文档以达到 30-60 条，每条 200+ 字符
        base_documents = [
            "A middle-aged man is sitting comfortably and eating a large bowl of homemade pasta in a quiet, traditional Italian restaurant located in the heart of Rome, enjoying the fine red wine and the peaceful atmosphere of the evening while watching local people pass by outside the window.",
            "In a small, family-owned bakery on the bustling corner of the main street, a young man is quickly eating a large, freshly baked piece of sourdough bread that smells wonderful and has a perfectly golden, crunchy crust that he spent several minutes admiring before taking his first bite.",
            "A young girl wearing a bright pink dress and white shoes is carefully carrying a small, sleeping baby wrapped in a soft, hand-knitted white blanket across the sunny park, making sure to step very slowly and avoid any small stones or uneven patches on the narrow dirt path.",
            "Under the golden afternoon sun of a warm summer day, a middle-aged man is confidently riding a powerful, deep brown horse along the edge of a crystal clear lake, feeling the cool breeze on his face and listening to the rhythmic, satisfying sound of hooves hitting the soft ground.",
            "The extremely talented woman is playing a beautiful and highly complex melody on her vintage, well-maintained violin in front of a large, captivated audience at the national concert hall during the highly anticipated annual summer music festival that draws people from all over the country.",
            "Two elderly people wearing heavy, warm winter coats and colorful scarves are slowly walking down the busy city street, holding hands tightly and talking softly about their shared memories of how the city looked many decades ago when they were young and just starting their lives together.",
            "A large black guard dog in the front yard is barking very loudly and aggressively at a total stranger who is standing cautiously near the locked iron gate, trying to deliver a large, heavy parcel that contains fragile glassware and requires a physical signature for successful delivery.",
            "The sun is shining extremely brightly today in the perfectly clear blue sky, warming the ground and making the colorful flowers in the public garden bloom beautifully while various types of insects buzz around them in the warm air, creating a natural symphony of sounds and colors.",
            "I absolutely love programming in Python because it allows me to build complex, scalable applications very quickly with its clean, readable syntax and a vast, supportive ecosystem of libraries that provide efficient solutions for almost any technical problem I encounter in my daily work.",
            "Artificial intelligence is rapidly and fundamentally changing the entire world by automating routine human tasks, significantly improving medical diagnoses through advanced deep learning algorithms, and creating entirely new economic opportunities in almost every industry imaginable today."
        ]
        
        # 构造 60 条文档
        documents = []
        for i in range(6):
            for doc in base_documents:
                documents.append(f"[{i*10 + base_documents.index(doc)}] {doc}")

        print(f"待处理文档总数: {len(documents)}")
        print(f"首条文档长度: {len(documents[0])} 字符")

        # 调用 rerank API
        rerank_url = f"{base_url}/rerank"
        payload = {"model": rerank_model_id, "query": query, "documents": documents}

        import time
        start_time = time.time()
        # 增加客户端超时到 300s
        response = requests.post(rerank_url, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        end_time = time.time()

        print(f"\n⏱️ Rerank 耗时: {end_time - start_time:.2f} 秒")
        print("\n📊 Rerank 结果 (Top 5):")
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
