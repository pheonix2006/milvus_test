import requests
import json
import os

# ================= 配置区 =================
# OpenAI 兼容 API Key
api_key = os.getenv("AZURE_OPENAI_API_KEY", "your-api-key-here")

# API 完整地址（包含 /chat/completions 路径）
base_url = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com/...")

# 使用的模型
model = os.getenv("AZURE_OPENAI_MODEL", "gpt-5.1")

# API 类型（必需）
# standard: 标准 OpenAI 格式（使用 messages 输入，choices 输出）
# custom_azure: 特殊 Azure 代理格式（使用 input 输入，output 输出）
api_type = "custom_azure"
# ==========================================

def call_api(prompt: str):
    """
    封装 API 调用逻辑
    """
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key  # Azure 风格请求头
    }
    
    # 也可以备选 Authorization 头以增加兼容性
    # headers["Authorization"] = f"Bearer {api_key}"

    if api_type == "custom_azure":
        # 按照用户说明：使用 input 输入
        payload = {
            "input": prompt,
            "model": model
        }
    else:
        # Standard OpenAI 格式
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

    print(f"正在向 {base_url} 发送请求...")
    print(f"请求载荷: {json.dumps(payload, ensure_ascii=False)}")

    try:
        response = requests.post(
            base_url, 
            headers=headers, 
            json=payload,
            timeout=60
        )
        
        # 检查 HTTP 状态码
        response.raise_for_status()
        
        result = response.json()
        
        # 获取结果
        if api_type == "custom_azure":
            # 按照用户说明：使用 output 输出
            answer = result.get("output", "错误：未在返回结果中找到 'output' 字段")
        else:
            # 标准 OpenAI 格式解析
            try:
                answer = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                answer = f"错误：解析标准格式失败，原始响应: {result}"
        
        return answer

    except requests.exceptions.HTTPError as e:
        return f"HTTP 错误: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"发生异常: {str(e)}"

if __name__ == "__main__":
    test_prompt = "你好，请确认你是否能收到这条测试消息。"
    print(f"\n--- 测试开始 ---\n")
    print(f"Prompt: {test_prompt}\n")
    
    response_text = call_api(test_prompt)
    
    print(f"\n--- 获取到的结果 ---\n")
    print(response_text)
    print(f"\n--- 测试结束 ---")
