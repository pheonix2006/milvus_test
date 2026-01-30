import requests
import json

def test_chat_api():
    # FastAPI 运行的默认地址
    url = "http://127.0.0.1:8000/chat"
    
    # 构造请求数据
    payload = {
        "prompt": "请用一句话介绍一下什么是向量数据库。",
        "model": "gpt-5.1",
        "system_prompt": "你是一个严谨的计算机科学家。"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("--- 正在测试 api.py 中的 /chat 接口 ---")
    print(f"请求地址: {url}")
    print(f"发送内容: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    
    try:
        # 发送 POST 请求到我们刚刚修改的 FastAPI 服务
        # 注意：运行此脚本前，请确保已经启动了 FastAPI 服务：
        # python src/milvus_test/api.py
        response = requests.post(url, headers=headers, json=payload, timeout=65)
        
        # 检查是否成功
        if response.status_code == 200:
            result = response.json()
            print("\n--- 接口调用成功 ---")
            print(f"模型回答: {result.get('answer')}")
        else:
            print(f"\n--- 接口调用失败 ---")
            print(f"状态码: {response.status_code}")
            print(f"错误详情: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n错误: 无法连接到服务器。请确保你已经运行了 'python src/milvus_test/api.py'")
    except Exception as e:
        print(f"\n发生意外错误: {e}")

if __name__ == "__main__":
    test_chat_api()
