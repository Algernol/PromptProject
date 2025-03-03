import os
import dashscope

from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# 设置 Hugging Face Hub 的镜像地址
os.environ['HUGGINGFACE_HUB_URL'] = 'https://hf-mirror.com'

# 设置通义千问API
dashscope.api_key = "sk-bb771fa94f504363b818308c6f6d779d"

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

while True:
    message = input('user:')
    if message == "结束对话":
        break
    print("正在检索中.....")
    similarDocs = db.similarity_search(message, k=10)
    summary_prompt = "".join([doc.page_content for doc in similarDocs])
    send_message = f"这是修改问题的参考文档({summary_prompt})，这里是文档所需的要求({message})，这是完整问题（{content}），请注意：请你帮我将修改后的问题返还给我，不要说除了问题之外的任何话，只需要返回问题，不要省略其他点的坐标。"
    messages.append({'role': Role.USER, 'content': send_message})
    whole_message = ''
    # 切换模型
    responses = Generation.call(Generation.Models.qwen_max, messages=messages, result_format='message', stream=True,
                                incremental_output=True)
    print("检索完毕")
    print('system:', end='')
    for response in responses:
        whole_message += response.output.choices[0]['message']['content']
        print(response.output.choices[0]['message']['content'], end='')
    print()
    messages.append({'role': 'assistant', 'content': whole_message})

