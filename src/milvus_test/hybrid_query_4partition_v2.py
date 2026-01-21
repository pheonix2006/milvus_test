import os
import json
import time
from pathlib import Path
import numpy as np
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,host.docker.internal'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'
from scipy.sparse import csr_matrix
from pymilvus import (
    connections,
    Collection,
    AnnSearchRequest,
    RRFRanker,
    WeightedRanker
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# ==========================================
# 0. 环境与配置
# ==========================================


URI = "http://localhost:19530"
TOKEN = "root:Milvus"
COLLECTION_NAME = "hybrid_rag_collection_v1"  # 必须与入库时的名称一致

# 加载配置
def load_local_config():
    config_path = Path(__file__).resolve().parent / "local_config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

local_cfg = load_local_config()
hf_token = os.getenv("HF_TOKEN") or local_cfg.get("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

class CSRWithLen(csr_matrix):
    """解决 Scipy 稀疏矩阵长度歧义问题 (Query 阶段同样需要)"""
    def __len__(self):
        return self.shape[0]

# ==========================================
# 1. Router 策略定义 (映射到最新的物理分区)
# ==========================================
# Key 对应查询意图，Value 对应 Milvus 物理分区名
STRATEGY_MAP = {
    "diesel": ["partition_diesel"],                   # 柴油机
    "gas_15n": ["partition_Natural_gas_15N"],         # 15N 燃气机
    "gas_12n": ["partition_Natural_gas_12N"],         # 12N 燃气机
    "gas_general": ["partition_Natural_gas_General_knowledge"], # 燃气机通用知识
    "global": []  # 全局搜索
}

def get_strategy_partitions(intent: str):
    """根据意图获取目标分区"""
    return STRATEGY_MAP.get(intent, [])  # 默认全局搜索

# ==========================================
# 2. 检索执行层
# ==========================================
class RouterRetriever:
    def __init__(self, uri, token, collection_name):
        print(f"正在连接 Milvus: {uri}...")
        connections.connect(uri=uri, token=token)
        self.col = Collection(collection_name)
        
        print(f"正在加载计划中的集合 {collection_name} 到内存...")
        self.col.load()
        
        print("正在加载 BGE-M3 模型...")
        # 建议根据硬件调整 device
        self.ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        print("检索器初始化完成。")

    def search(self, query_text: str, intent: str = "global", top_k: int = 5):
        """
        执行混合检索：结合稠密与稀疏向量，并支持分区过滤
        """
        # 1. 获取目标分区
        target_partitions = get_strategy_partitions(intent)
        print(f"\n[Router] 收到查询: '{query_text}'")
        print(f"[Router] 判定意图: '{intent}' -> 锁定分区: {target_partitions if target_partitions else 'ALL (Global Search)'}")

        # 2. 向量化 (BGE-M3)
        start_time = time.time()
        embeddings = self.ef([query_text])
        
        # 转换格式
        query_dense = embeddings["dense"] # list of dense vectors
        query_sparse = CSRWithLen(embeddings["sparse"]).tocsr()

        # 3. 构建混合检索请求 (AnnSearchRequest) 
        
        # 3.1 稠密请求 (Dense / HNSW)
        limit = top_k * 2
        dense_req = AnnSearchRequest(
            data=query_dense,
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"ef": max(100, limit)}},
            limit=limit
        )

        # 3.2 稀疏请求 (Sparse / WAND)
        sparse_req = AnnSearchRequest(
            data=[query_sparse],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.1}}, 
            limit=limit
        )

        # 4. 执行 Hybrid Search (使用 RRF 合并召回)
        # k=60 是 RRF 的标准常量
        rerank = RRFRanker(k=60) 
        
        res = self.col.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=rerank,
            limit=top_k,
            partition_names=target_partitions if target_partitions else None,
            output_fields=["content", "metadata"]
        )
        
        latency = (time.time() - start_time) * 1000
        return res[0], latency

# ==========================================
# 3. 模拟测试
# ==========================================
if __name__ == "__main__":
    retriever = RouterRetriever(URI, TOKEN, COLLECTION_NAME)

    # 场景示例 1: 定向搜索 柴油/Diesel 分区
    q1 = "柴油发动机故障诊断建议有哪些？"
    hits, lat = retriever.search(q1, intent="diesel", top_k=3)
    
    print(f"\n--- Diesel 分区搜索结果 (耗时: {lat:.2f}ms) ---")
    for i, hit in enumerate(hits):
        meta = hit.entity.get("metadata")
        print(f"{i+1}. [Score: {hit.score:.4f}] 分区: {meta.get('partition')} | 内容预览: {hit.entity.get('content')[:100]}...")

    # 场景示例 2: 全局搜索
    q2 = "天然气发动机 15N 与 12N 的主要区别是什么？"
    hits, lat = retriever.search(q2, intent="global", top_k=3)
    
    print(f"\n--- 全局搜索结果 (耗时: {lat:.2f}ms) ---")
    for i, hit in enumerate(hits):
        meta = hit.entity.get("metadata")
        print(f"{i+1}. [Score: {hit.score:.4f}] 分区: {meta.get('partition')} | 内容预览: {hit.entity.get('content')[:100]}...")

    # 场景示例 3: 15N 燃气机定向搜索
    q3 = "15N 发动机的 OBD 故障码 P0300 应该如何处理？"
    hits, lat = retriever.search(q3, intent="gas_15n", top_k=3)
    
    print(f"\n--- 15N 燃气机分区搜索结果 (耗时: {lat:.2f}ms) ---")
    for i, hit in enumerate(hits):
        meta = hit.entity.get("metadata")
        print(f"{i+1}. [Score: {hit.score:.4f}] 分区: {meta.get('partition')} | 内容预览: {hit.entity.get('content')[:100]}...")
