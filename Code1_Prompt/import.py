from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 导入文本
loader = UnstructuredLoader("local_database/vrp.txt")
data = loader.load()

print(data)

# 文本切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
split_docs = text_splitter.split_documents(data)

# 获取切分文档数量
document_count = len(split_docs)
# print(split_docs)

model_name = "bce-embedding-base_v1"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 修改元数据，将 languages 列表转换为字符串
for doc in split_docs:
    if isinstance(doc.metadata['languages'], list) and len(doc.metadata['languages']) > 0:
        doc.metadata['languages'] = doc.metadata['languages'][0]  # 取列表的第一个元素

# 初始化 Chroma 数据库
db = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma/news_test")


