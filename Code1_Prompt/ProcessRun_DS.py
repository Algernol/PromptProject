import json
import os
import dashscope

import requests  # 新增请求库
# 删除 dashscope 相关导入
# 新增 ollama 服务地址配置
##服务器 http://10.200.125.118:11434
OLLAMA_API = "http://10.200.125.118:11434/api/generate"
##OLLAMA_API = "http://localhost:11434/api/generate"
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma



def deepseek_generate(prompt, context=""):
    """调用本地 DeepSeek 模型的生成函数"""
    full_prompt = f"{context}\n\n用户需求：{prompt}\n请严格按照要求处理问题："  # 优化后的提示模板

    data = {
        "model": "deepseek-r1:32b",
        "prompt": full_prompt,
        "stream": True,  # 保持流式输出
        # "options": {
        #     "temperature": 0.3,  # 降低随机性保证输出稳定性
        #     "max_tokens": 4096,
        #     "stop": ["\n\n"]  # 停止序列防止多余输出
        # }
    }

    try:
        response = requests.post(
            OLLAMA_API,
            json=data,
            stream=True,
            timeout=60  # 适当延长超时时间
        )
        response.raise_for_status()
        # response.json()["response"]
        # print(response.json()["response"])

        # 处理流式响应
        # full_response = ""
        # for line in response.iter_lines():
        #     if line:
        #         chunk = json.loads(line.decode("utf-8"))
        #         if not chunk["done"]:
        #             content = chunk["response"]
        #             full_response += content
        #             print(content, end="", flush=True)
        # return full_response
        # 修改响应处理部分
        full_response = ""
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        if chunk.get("done", False):
                            break  # 正确识别结束标记
                        content = chunk.get("response", "")
                        full_response += content
                        print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        continue  # 跳过无效数据包
        except requests.exceptions.ChunkedEncodingError as e:
            print(f"\n流数据接收异常: {str(e)}")

        return full_response

    except Exception as e:
        print(f"\n模型调用失败：{str(e)}")
        return ""


# 记录上下文
messages = []

model_name = "bce-embedding-base_v1"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 对数据进行加载
db = Chroma(persist_directory="./chroma/news_test", embedding_function=embeddings)

# 打开对应的问题文件
with open('data/X-n439-k37.vrp', 'r') as file:
    content = file.read()

    # 打开对应的问题文件
with open('local_database/vrp.txt', 'r',encoding="utf-8") as file:
    explain = file.read()


while True:
    message = input('user:')
    if message == "结束对话":
        break
    print("  正在检索中.....")
    similarDocs = db.similarity_search(message, k=10)
    summary_prompt = "".join([doc.page_content for doc in similarDocs])



    # 重构提示工程模板
    send_message = f"""
    根据以下参考文档和需求修改问题：

    [参考文档]
    {summary_prompt}

    [原始问题]
    {content}

    [用户需求]
    {message}

    请严格遵循：
    1. 按照要求修改原始问题的内容
    2. 保留所有坐标点数据
    3. 使用原始问题数据格式
    4. 不要添加任何解释
    """

    # send_message = f"""
    # 根据以下参考文档回答问题：
    #
    # [参考文档]
    # {summary_prompt}
    #
    # [原始问题]
    # {content}
    #
    # [用户需求]
    # {message}
    #
    # 数据部分前三个数字是什么
    #
    # """

    print("检索完毕")
    print('system:', end='')
    response = deepseek_generate(send_message)  # 调用改造后的生成函数
    print()
    messages.append({'role': 'user', 'content': message})
    messages.append({'role': 'assistant', 'content': response})

