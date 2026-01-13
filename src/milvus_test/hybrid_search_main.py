import os
from pathlib import Path

# ==========================================
# 0. 代理与镜像设置 (必须在导入 pymilvus/transformers 前设置)
# ==========================================
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,host.docker.internal'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'

import time
import json
import requests
import numpy as np
from scipy.sparse import csr_matrix
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
    MilvusClient
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


class CSRWithLen(csr_matrix):
    # Avoid scipy sparse array length ambiguity
    def __len__(self):  # type: ignore[override]
        return self.shape[0]


def load_local_config():
    config_path = Path(__file__).resolve().parent / "local_config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# 读取本地配置文件或环境变量，避免在代码中硬编码密钥
local_cfg = load_local_config()
hf_token = os.getenv("HF_TOKEN") or local_cfg.get("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
else:
    print("[WARN] HF_TOKEN 未设置：如模型需要鉴权，请在环境变量或 local_config.json 中配置")

# ==========================================
# 1. 初始化 BGE-M3 模型
# ==========================================
print("正在加载 BGE-M3 模型 (可能会下载模型权重)...")
# use_fp16=False if on CPU, True if on GPU
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
dense_dim = ef.dim["dense"]
print(f"BGE-M3 模型加载完成。稠密向量维度: {dense_dim}")

# ==========================================
# 2. Milvus 连接与 Schema 设置
# ==========================================
URI = "http://localhost:19530"
TOKEN = "root:Milvus"
collection_name = "hybrid_search_collection"

print(f"正在连接到 Milvus: {URI}...")
connections.connect(uri=URI, token=TOKEN)

if utility.has_collection(collection_name):
    print(f"删除旧集合: {collection_name}")
    utility.drop_collection(collection_name)

# 定义 Schema
fields = [
    # 使用自动生成的主键
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
    # 原始文本
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
    # 类别标签 (可选)
    FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=100),
    # 稀疏向量 (用于关键词匹配)
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    # 稠密向量 (用于语义搜索)
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
]
schema = CollectionSchema(fields, description="Hybrid search collection with dense and sparse vectors")

print(f"正在创建集合: {collection_name}...")
col = Collection(collection_name, schema)

# 创建索引
print("正在为向量字段创建索引...")
sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
col.create_index("sparse_vector", sparse_index)

dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
col.create_index("dense_vector", dense_index)

col.load()
print("集合加载完成。")

# ==========================================
# 3. 准备数据并插入
# ==========================================
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

print("正在生成稠密与稀疏向量...")
embeddings = ef(docs)

# 稀疏向量：coo_array -> csr_matrix(float32) with explicit __len__
sparse_vectors = CSRWithLen(embeddings["sparse"]).tocsr().astype(np.float32)
# 稠密向量：list/ndarray -> ndarray(float32)
dense_vectors = np.asarray(embeddings["dense"], dtype=np.float32, order="C")

# 插入数据
print("正在插入数据...")
entities = [
    docs,                                  # text
    ["history"] * len(docs),               # subject (示例)
    sparse_vectors,                        # sparse_vector
    dense_vectors                          # dense_vector
]
col.insert(entities)
col.flush()
print(f"数据插入成功。当前实体总数: {col.num_entities}")

# ==========================================
# 4. 混合检索逻辑
# ==========================================
def hybrid_retrieval(query_text, top_k=10):
    query_embeddings = ef([query_text])
    query_sparse = CSRWithLen(query_embeddings["sparse"]).tocsr().astype(np.float32)
    query_dense = np.asarray(query_embeddings["dense"], dtype=np.float32, order="C")
    
    # 稠密检索请求
    dense_search_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest(
        query_dense, 
        "dense_vector", 
        dense_search_params, 
        limit=top_k
    )
    
    # 稀疏检索请求
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        query_sparse, 
        "sparse_vector", 
        sparse_search_params, 
        limit=top_k
    )
    
    # 使用加权排名 (这里暂时均等权重，主要靠后面的 Rerank)
    rerank_strategy = WeightedRanker(0.5, 0.5)
    
    print(f"\n正在发起混合检索: '{query_text}'")
    res = col.hybrid_search(
        [sparse_req, dense_req], 
        rerank=rerank_strategy, 
        limit=top_k, 
        output_fields=["text"]
    )
    return res[0]

# ==========================================
# 5. 调用自定义 Rerank API
# ==========================================
def custom_rerank(query, milvus_hits, host="http://localhost:9997/v1"):
    if not milvus_hits:
        return []
        
    documents = [hit.get("text") for hit in milvus_hits]
    
    # 获取可用的 rerank 模型 ID (参考 rerank_test.py)
    try:
        models_res = requests.get(f"{host}/models", timeout=5)
        models_res.raise_for_status()
        models_data = models_res.json().get("data", [])
        model_id = next((m["id"] for m in models_data if "rerank" in m["id"].lower()), None)
        if not model_id:
            print("未找到可用的 Rerank 模型")
            return None
    except Exception as e:
        print(f"获取 Rerank 模型列表失败: {e}")
        return None

    print(f"使用 Rerank 模型: {model_id} 进行精排...")
    url = f"{host}/rerank"
    payload = {
        "model": model_id,
        "query": query,
        "documents": documents,
        "top_n": 5
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        rerank_data = response.json()
        
        results = []
        for item in rerank_data.get('results', []):
            idx = item['index']
            results.append({
                "text": documents[idx],
                "score": item.get('relevance_score') or item.get('score', 0)
            })
        return results
    except Exception as e:
        print(f"Rerank API 调用失败: {e}")
        return None

# ==========================================
# 6. 主流程测试
# ==========================================
if __name__ == "__main__":
    query = "Who is Alan Turing and what did he do for AI?"
    
    # 第 1 步: 混合检索
    hits = hybrid_retrieval(query, top_k=10)
    
    print("\n--- 混合检索 (召回) 结果 ---")
    for i, hit in enumerate(hits):
        print(f"{i+1}. [Score: {hit.score:.4f}] {hit.get('text')}")
        
    # 第 2 步: Rerank 精排
    final_results = custom_rerank(query, hits)
    
    if final_results:
        print("\n--- Rerank 精排后的 Top 5 ---")
        for i, res in enumerate(final_results):
            print(f"{i+1}. [Rerank Score: {res['score']:.4f}] {res['text']}")
    else:
        print("\n未能获取 Rerank 结果，请检查后端服务是否启动。")
