import os
import json
import time
from pathlib import Path
from typing import List, Dict
import numpy as np
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,host.docker.internal'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'
from scipy.sparse import csr_matrix
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


URI = "http://localhost:19530"
TOKEN = "root:Milvus"

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
# 1. 数据库初始化
# ==========================================
def init_collection_and_partitions(uri, token, col_name, partition_names):
    print(f"正在连接 Milvus: {uri}...")
    connections.connect(uri=uri, token=token)

    if utility.has_collection(col_name):
        print(f"[Warn] 集合 {col_name} 已存在。为了确保 Schema 一致，正在删除重建...")
        utility.drop_collection(col_name)

    print(f"创建集合: {col_name}...")
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    ]
    
    schema = CollectionSchema(fields, description="Unified Hybrid RAG Collection")
    col = Collection(col_name, schema)

    for p_name in partition_names:
        if not col.has_partition(p_name):
            print(f"  -> 创建物理分区: {p_name}")
            col.create_partition(p_name)
    
    print("正在创建索引...")
    sparse_index_params = {
        "index_type": "SPARSE_WAND",
        "metric_type": "IP",
        "params": {"drop_ratio_build": 0.2}
    }
    col.create_index("sparse_vector", sparse_index_params)

    dense_index_params = {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 200}
    }
    col.create_index("dense_vector", dense_index_params)
    return col

def load_docs(path: Path, separator: str) -> List[str]:
    raw = path.read_text(encoding="utf-8")
    parts = raw.split(separator)
    docs = [p.strip() for p in parts if p.strip()]
    return docs

def process_file(file_path: Path, partition_name: str, col: Collection, ef_model, batch_size=50):
    print(f"\n处理文件: {file_path.name} -> 目标分区: {partition_name}")
    docs = load_docs(file_path, "# separator")
    
    if not docs:
        print("  -> 文件为空或无有效文档块，跳过")
        return

    print(f"  -> 包含 {len(docs)} 个文档块，开始向量化与入库...")
    
    total_inserted = 0
    for i in range(0, len(docs), batch_size):
        batch_texts = docs[i : i + batch_size]
        embeddings = ef_model(batch_texts)
        dense_vectors = embeddings["dense"]
        sparse_vectors = CSRWithLen(embeddings["sparse"]).tocsr()

        metadatas = [{"source_file": file_path.name, "partition": partition_name} for _ in batch_texts]

        data = [batch_texts, metadatas, sparse_vectors, dense_vectors]
        col.insert(data, partition_name=partition_name)
        total_inserted += len(batch_texts)
        print(f"    -> 已插入 batch {i // batch_size + 1} (Total: {total_inserted})")

def main():
    config, active_dataset, dataset_cfg = load_config()
    
    hf_token = os.getenv("HF_TOKEN") or config.get("HF_TOKEN")
    if hf_token: os.environ["HF_TOKEN"] = hf_token

    print(f"--- 启动统一入库流程 (当前资料集: {active_dataset}) ---")
    
    print("正在加载 BGE-M3 模型...")
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    
    col_name = dataset_cfg["collection_name"]
    data_dir = Path(dataset_cfg["data_dir"])
    file_map = dataset_cfg["file_to_partition_map"]
    
    if not data_dir.exists():
        print(f"[Error] 数据目录 {data_dir} 不存在")
        return
    
    target_partitions = list(file_map.values())
    col = init_collection_and_partitions(URI, TOKEN, col_name, target_partitions)

    for file_name, partition_name in file_map.items():
        file_path = data_dir / file_name
        if file_path.exists():
            process_file(file_path, partition_name, col, ef)
        else:
            print(f"[Warn] 未找到文件: {file_name}，跳过")

    print("\n所有数据插入完成，正在 Flush...")
    col.flush()
    print("正在加载集合 (Load)...")
    col.load()
    print(f"\n成功! 资料集 '{active_dataset}' 已存入集合: {col_name}")

if __name__ == "__main__":
    main()
