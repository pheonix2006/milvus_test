import requests
import json

def test_milvus_api():
    url = "http://localhost:8000/search"
    
    # 测试数据
    payload = {
        "query": "How to use decorators in Python?",
        "top_k": 5,        # 缩小数量方便显示
        "strategy": "python"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"正在发送请求到: {url}")
    print(f"请求内容: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print("\n[成功] 响应结果:")
            # 现在返回结构是 {"pure_documents": [...]}
            results = data.get('pure_documents', [])
            print(f"召回片段数量: {len(results)}")
            
            for i, content in enumerate(results):
                preview = content[:100].replace('\n', ' ')
                print(f"{i+1}. {preview}...")
        else:
            print(f"\n[失败] 状态码: {response.status_code}")
            print(f"错误详情: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n[错误] 无法连接到服务器。请确保 API 服务已启动 (uv run python src/milvus_test/api.py)")
    except Exception as e:
        print(f"\n[错误] 发生异常: {e}")

if __name__ == "__main__":
    test_milvus_api()
