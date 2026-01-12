import os
import time
import requests
from pymilvus import MilvusClient
import ollama

# ==========================================
# 0. 代理设置 (防止全局代理影响本地服务连接)
# ==========================================
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,host.docker.internal'

# ==========================================
# 1. 连接设置
# ==========================================
print("正在连接到 Docker Milvus...")
client = MilvusClient(
    uri="http://localhost:19530", 
    token="root:Milvus" 
)
print("连接成功！")

# ==========================================
# 2. 准备 Collection
# ==========================================
collection_name = "demo_collection"
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

# 创建集合
client.create_collection(
    collection_name=collection_name,
    dimension=1024,
)
print(f"集合 {collection_name} 创建成功。")

# ==========================================
# 3. 准备数据与向量化
# ==========================================
class OllamaEmbeddingFunction:
    def __init__(self, model_name="mxbai-embed-large:latest", host="http://localhost:11434"):
        self.client = ollama.Client(host=host)
        self.model_name = model_name

    def encode_documents(self, docs):
        return [self.client.embeddings(model=self.model_name, prompt=doc)["embedding"] for doc in docs]

    def encode_queries(self, queries):
        return [self.client.embeddings(model=self.model_name, prompt=query)["embedding"] for query in queries]


embedding_fn = OllamaEmbeddingFunction(model_name="mxbai-embed-large:latest")

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
    "The Enigma machine was an encryption device used by Germany during WWII.",
    "Turing led the team at Bletchley Park that cracked the German Enigma code.",
    "The Turing Test is a measure of a machine's ability to exhibit intelligent behavior.",
    "Milvus is an open-source vector database built for AI applications.",
    "Vector databases are essential for storing and searching high-dimensional embeddings.",
    "RAG stands for Retrieval-Augmented Generation, a technique to improve LLM accuracy.",
    "Ollama allows you to run large language models locally on your machine.",
    "Python is the most popular programming language for data science and AI.",
    "Deep learning is a subset of machine learning based on artificial neural networks.",
    "Large language models like GPT-4 are trained on massive amounts of text data.",
    "Natural language processing helps computers understand and interpret human language.",
    "Information retrieval systems use similarity search to find relevant documents.",
]

vectors = embedding_fn.encode_documents(docs)
data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

# ==========================================
# 4. 插入数据
# ==========================================
client.insert(collection_name=collection_name, data=data)
time.sleep(1) 

# ==========================================
# 5. 语义搜索 (向量召回)
# ==========================================
query_text = "Who is Alan Turing?"
query_vectors = embedding_fn.encode_queries([query_text])

print(f"\n正在执行向量召回: '{query_text}'")
search_res = client.search(
    collection_name=collection_name,
    data=query_vectors,
    limit=10, 
    output_fields=["text", "subject"],
)

print("--- 向量召回结果 ---")
for hits in search_res:
    for hit in hits:
        print(f" - ID: {hit['id']}, 距离: {hit['distance']:.4f}, 内容: {hit['entity']['text']}")

# ==========================================
# 6. Rerank 重排序
# ==========================================
print("\n正在执行 Rerank (localhost:9997)...")

def get_rerank(query, docs, host="http://localhost:9997/v1"):
    # 自动获取可用的 rerank 模型 ID
    try:
        models_res = requests.get(f"{host}/models")
        models_res.raise_for_status()
        models_data = models_res.json().get("data", [])
        model_id = next((m["id"] for m in models_data if "rerank" in m["id"].lower()), None)
        if not model_id:
            print("未找到可用的 Rerank 模型")
            return None
    except Exception as e:
        print(f"获取模型列表失败: {e}")
        return None

    url = f"{host}/rerank"
    payload = {
        "model": model_id,
        "query": query,
        "documents": [doc['entity']['text'] for doc in docs],
        "top_n": 3
    }
    try:
        response = requests.post(url, json=payload, timeout=100, proxies={"http": None, "https": None})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Rerank 失败: {e}")
        return None

rerank_results = get_rerank(query_text, search_res[0])

if rerank_results and 'results' in rerank_results:
    print("--- Rerank 重排后的 Top 3 ---")
    for result in rerank_results['results']:
        idx = result['index']
        score = result.get('relevance_score') or result.get('score', 0)
        content = search_res[0][idx]['entity']['text']
        print(f" - 分数: {score:.4f}, 内容: {content}")
else:
    print("未能获取 Rerank 结果。")
