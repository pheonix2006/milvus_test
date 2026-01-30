import os
import json
import time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ==========================================
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



URI = os.getenv("MILVUS_URI", "http://localhost:19530")
TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")

def load_config():
    config_path = Path(__file__).resolve().parent / "local_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    active_dataset = config.get("active_dataset", "engine")
    dataset_cfg = config.get("datasets", {}).get(active_dataset, {})
    
    if not dataset_cfg:
        raise ValueError(f"Dataset configuration for '{active_dataset}' not found in local_config.json")
    
    return config, active_dataset, dataset_cfg

class CSRWithLen(csr_matrix):
    """解决 Scipy 稀疏矩阵长度歧义问题"""
    def __len__(self):
        return self.shape[0]

# ==========================================
# 1. 检索执行层
# ==========================================
class RouterRetriever:
    def __init__(self, uri, token):
        self.config, self.active_dataset, self.dataset_cfg = load_config()
        self.collection_name = self.dataset_cfg.get("collection_name")
        self.strategy_map = self.dataset_cfg.get("strategy_map", {})
        
        # 加载 HF Token
        hf_token = os.getenv("HF_TOKEN") or self.config.get("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        print(f"正在连接 Milvus: {uri} (当前资料集: {self.active_dataset})...")
        connections.connect(uri=uri, token=token)
        self.col = Collection(self.collection_name)
        
        print(f"正在加载计划中的集合 {self.collection_name} 到内存...")
        self.col.load()
        
        print("正在加载 BGE-M3 模型...")
        self.ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        print("检索器初始化完成。")

    def get_strategy_partitions(self, intent: str):
        """根据意图获取目标分区"""
        return self.strategy_map.get(intent, [])

    def search(self, query_text: str, intent: str = "global", top_k: int = 5):
        """
        执行混合检索：结合稠密与稀疏向量，并支持分区过滤
        """
        # 1. 获取目标分区
        target_partitions = self.get_strategy_partitions(intent)
        print(f"\n[Router] 收到查询: '{query_text}' (资料集: {self.active_dataset})")
        print(f"[Router] 判定意图: '{intent}' -> 锁定分区: {target_partitions if target_partitions else 'ALL (Global Search)'}")

        # 2. 向量化 (BGE-M3)
        start_time = time.time()
        embeddings = self.ef([query_text])
        
        # 转换格式
        query_dense = embeddings["dense"] 
        query_sparse = CSRWithLen(embeddings["sparse"]).tocsr()

        # 3. 构建混合检索请求
        limit = top_k * 2
        dense_req = AnnSearchRequest(
            data=query_dense,
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"ef": max(100, limit)}},
            limit=limit
        )

        sparse_req = AnnSearchRequest(
            data=[query_sparse],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.1}}, 
            limit=limit
        )

        # 4. 执行 Hybrid Search
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
# 2. 模拟测试
# ==========================================
if __name__ == "__main__":
    retriever = RouterRetriever(URI, TOKEN)
    
    # 根据当前配置进行简单的测试搜索
    test_query = "测试查询"
    hits, lat = retriever.search(test_query, intent="global", top_k=3)
    
    print(f"\n--- 检索结果 (耗时: {lat:.2f}ms) ---")
    for i, hit in enumerate(hits):
        meta = hit.entity.get("metadata")
        print(f"{i+1}. [Score: {hit.score:.4f}] 分区: {meta.get('partition')} | 内容预览: {hit.entity.get('content')[:100]}...")
