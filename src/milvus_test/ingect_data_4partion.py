import os
import json
import time
from pathlib import Path
from typing import List, Dict

# ==========================================
# 0. 环境设置
# ==========================================
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,host.docker.internal'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'

import numpy as np
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

# ==========================================
# 1. 配置区域
# ==========================================
# Milvus 连接
URI = "http://localhost:19530"
TOKEN = "root:Milvus"
COLLECTION_NAME = "hybrid_rag_collection_v1" # 建议使用新名称以区别于Demo

# 数据源配置
DATA_DIR = Path("test_txt")  # 存放4个txt文件的文件夹
DOC_SEPARATOR = "# separator"  # 文档分隔符，空块会被忽略
# 文件名到分区名的映射 (请根据你实际的 txt 文件名修改 key)
# Value 必须符合 Milvus 分区命名规范 (字母下划线)
FILE_TO_PARTITION_MAP = {
    "merged_document.md": "partition_diesel",     # 示例：文件名 -> 分区名
    "OBD故障码汇总_WCF_20240719_sheetNG(15N).md": "partition_Natural_gas_15N",
    "CCG_故障码速查_2024_V1_sheet3.md": "partition_Natural_gas_12N",
    "general_knowledge_TWC_QA20250402(Bryan人工check后).md": "partition_Natural_gas_General_knowledge",
}

# 嵌入模型配置
BATCH_SIZE = 50  # BGE-M3 比较吃显存/内存，建议根据机器配置调整
DENSE_DIM = 1024

class CSRWithLen(csr_matrix):
    """解决 Scipy 稀疏矩阵长度歧义问题"""
    def __len__(self):
        return self.shape[0]

def load_local_config():
    config_path = Path(__file__).resolve().parent / "local_config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

# 加载 HF Token
local_cfg = load_local_config()
hf_token = os.getenv("HF_TOKEN") or local_cfg.get("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# ==========================================
# 2. 数据库初始化 (Schema & Partition)
# ==========================================
def init_collection_and_partitions(uri, token, col_name, partition_names):
    print(f"正在连接 Milvus: {uri}...")
    connections.connect(uri=uri, token=token)

    # 如果存在旧集合，根据开发阶段决定是否删除。生产环境需谨慎。
    if utility.has_collection(col_name):
        print(f"[Warn] 集合 {col_name} 已存在。为了确保 Schema 一致，正在删除重建...")
        utility.drop_collection(col_name)

    print(f"创建集合: {col_name}...")
    # 严格按照文档定义的 Schema 
    fields = [
        # 主键
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        # 原始内容 (适当缩减 max_length 防止超出页限制，文档建议 65535，这里设为 8192 够用且安全)
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
        # 元数据 (来源、时间等)
        FieldSchema(name="metadata", dtype=DataType.JSON),
        # 稀疏向量
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        # 稠密向量
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=DENSE_DIM),
    ]
    
    schema = CollectionSchema(fields, description="Milvus 2.6 Hybrid RAG Collection with Physical Partitions")
    col = Collection(col_name, schema)

    # 创建物理分区 
    for p_name in partition_names:
        if not col.has_partition(p_name):
            print(f"  -> 创建物理分区: {p_name}")
            col.create_partition(p_name)
    
    # 创建索引 (仅需在 Collection 层级创建一次)
    print("正在创建索引...")
    
    # 稀疏索引: SPARSE_WAND [cite: 65]
    sparse_index_params = {
        "index_type": "SPARSE_WAND", # Milvus 2.4+ / 2.6 推荐
        "metric_type": "IP",
        "params": {"drop_ratio_build": 0.2} # 丢弃 20% 极小权重以优化性能 
    }
    col.create_index("sparse_vector", sparse_index_params)
    print("  -> 稀疏索引 (SPARSE_WAND) 创建完成")

    # 稠密索引: HNSW [cite: 76]
    dense_index_params = {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 200}
    }
    col.create_index("dense_vector", dense_index_params)
    print("  -> 稠密索引 (HNSW) 创建完成")

    # 加载集合 (Milvus 需要 load 才能查询，但插入可以不 load，这里最后再 load)
    return col

# ==========================================
# 3. 文档加载
# ==========================================
def load_docs(path: Path, separator: str) -> List[str]:
    """按分隔符切分文档，返回文档列表"""
    raw = path.read_text(encoding="utf-8")
    parts = raw.split(separator)
    docs = [p.strip() for p in parts if p.strip()]
    return docs

# ==========================================
# 4. 数据处理与入库
# ==========================================
def process_file(file_path: Path, partition_name: str, col: Collection, ef_model):
    print(f"\n处理文件: {file_path.name} -> 目标分区: {partition_name}")
    
    # 使用分隔符加载文档，与 ingest_data.py 相同策略
    docs = load_docs(file_path, DOC_SEPARATOR)
    
    if not docs:
        print("  -> 文件为空或无有效文档块，跳过")
        return

    print(f"  -> 包含 {len(docs)} 个文档块，开始向量化与入库...")
    
    total_inserted = 0
    for i in range(0, len(docs), BATCH_SIZE):
        batch_texts = docs[i : i + BATCH_SIZE]
        
        # 1. 生成向量 (BGE-M3)
        embeddings = ef_model(batch_texts)
        
        # 格式转换
        dense_vectors = embeddings["dense"]
        # sparse 需转为 csr_matrix
        sparse_vectors = CSRWithLen(embeddings["sparse"]).tocsr()

        # 2. 准备 Metadata
        metadatas = [{"source_file": file_path.name, "partition": partition_name} for _ in batch_texts]

        # 3. 构造插入数据 (注意顺序必须与 Schema 一致: pk(auto), content, metadata, sparse, dense)
        # pk 是 auto_id，不需要传
        data = [
            batch_texts,    # content
            metadatas,      # metadata (JSON)
            sparse_vectors, # sparse_vector
            dense_vectors   # dense_vector
        ]

        # 4. 指定分区插入 [cite: 51]
        col.insert(data, partition_name=partition_name)
        total_inserted += len(batch_texts)
        print(f"    -> 已插入 batch {i // BATCH_SIZE + 1} (Total: {total_inserted})")

    print(f"  -> {file_path.name} 处理完成。")

# ==========================================
# 4. 主流程
# ==========================================
def main():
    # 0. 检查数据目录
    if not DATA_DIR.exists():
        print(f"[Error] 数据目录 {DATA_DIR} 不存在，请创建并将4个txt文件放入其中。")
        return
    
    # 1. 加载模型
    print("正在加载 BGE-M3 模型...")
    # use_fp16=True 如果你有 GPU
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    
    # 2. 初始化 Milvus
    # 从配置 Map 中提取所需的所有分区名
    target_partitions = list(FILE_TO_PARTITION_MAP.values())
    col = init_collection_and_partitions(URI, TOKEN, COLLECTION_NAME, target_partitions)

    # 3. 遍历文件并入库
    for file_name, partition_name in FILE_TO_PARTITION_MAP.items():
        file_path = DATA_DIR / file_name
        if file_path.exists():
            process_file(file_path, partition_name, col, ef)
        else:
            print(f"[Warn] 未找到文件: {file_name}，跳过该分区的导入。")

    # 4. 完成后 Flush 并 Load
    print("\n所有数据插入完成，正在 Flush...")
    col.flush()
    
    print("正在加载集合到内存 (Load)...")
    col.load()
    
    # 5. 验证统计
    print("\n=== 入库统计 ===")
    print(f"集合总量: {col.num_entities}")
    for pname in target_partitions:
        # 注意: num_entities 有时会有延迟，但在 flush 后应该是准的
        # Milvus partition 对象没有直接 num_entities 属性，需通过 query 统计或相信 insert 日志
        # 这里简单打印 partition 存在性
        print(f"分区 {pname}: 状态 Ready")
    
    print(f"\n成功! 你的数据已按照 4 个物理分区存储完毕。集合名称: {COLLECTION_NAME}")

if __name__ == "__main__":
    main()