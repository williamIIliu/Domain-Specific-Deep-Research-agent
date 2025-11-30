"""
测试 sglang 部署的模型服务
使用 OpenAI 兼容的 API 进行调用
"""

import requests
import json

# 服务配置
BASE_URL = "http://0.0.0.0:8060"
API_KEY = "sk-123456"
MODEL_NAME = "reward_model"


def test_chat_completion():
    """测试 Chat Completion API"""
    url = f"{BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "你好，请介绍一下你自己。"}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("=== Chat Completion 测试成功 ===")
        print(f"Response: {result['choices'][0]['message']['content']}")
        print(f"Usage: {result['usage']}")
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)


def test_with_openai_client():
    """使用 OpenAI SDK 调用"""
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url=f"{BASE_URL}/v1",
            api_key=API_KEY
        )
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "1+1等于多少？"}
            ],
            max_tokens=256
        )
        
        print("=== OpenAI SDK 测试成功 ===")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Usage: {response.usage}")
        
    except ImportError:
        print("未安装 openai 库，跳过 SDK 测试")
        print("安装命令: pip install openai")


def test_health():
    """检查服务健康状态"""
    url = f"{BASE_URL}/health"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print("=== 服务健康检查通过 ===")
            return True
        else:
            print(f"服务异常: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("无法连接到服务，请确认服务已启动")
        return False


def test_models():
    """列出可用模型"""
    url = f"{BASE_URL}/v1/models"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        models = response.json()
        print("=== 可用模型列表 ===")
        for model in models.get("data", []):
            print(f"  - {model['id']}")
    else:
        print(f"获取模型列表失败: {response.status_code}")


if __name__ == "__main__":
    print("开始测试 sglang 服务...\n")
    
    # 1. 健康检查
    if not test_health():
        exit(1)
    print()
    
    # 2. 列出模型
    test_models()
    print()
    
    # 3. 测试 Chat Completion
    test_chat_completion()
    print()
    
    # 4. 测试 OpenAI SDK
    test_with_openai_client()
