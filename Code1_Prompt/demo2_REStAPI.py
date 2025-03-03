import requests

def ollama_generate(prompt, model="deepseek-r1:7b"):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}  # 明确指定请求头
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # 主动触发 HTTP 错误
        return response.json()["response"]
    except requests.exceptions.HTTPError as e:
        return f"HTTP 错误: {e.response.text}"
    except Exception as e:
        return f"其他错误: {str(e)}"

print(ollama_generate("我想了解VRP问题以及优化思路"))