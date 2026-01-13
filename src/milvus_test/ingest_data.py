import os
from pathlib import Path

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


class CSRWithLen(csr_matrix):
    # Avoid scipy's default "length is ambiguous" behavior
    def __len__(self):  # type: ignore[override]
        return self.shape[0]


def ensure_collection(col_name: str, dense_dim: int, uri: str, token: str):
    connections.connect(uri=uri, token=token)

    if utility.has_collection(col_name):
        print(f"Collection {col_name} already exists; reusing it")
        return Collection(col_name)

    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    ]
    schema = CollectionSchema(fields, description="Hybrid search collection with dense and sparse vectors")
    col = Collection(col_name, schema)

    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    col.create_index("sparse_vector", sparse_index)
    dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    col.create_index("dense_vector", dense_index)

    col.load()
    return col


def load_docs(path: Path, separator: str) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    parts = raw.split(separator)
    docs = [p.strip() for p in parts if p.strip()]
    return docs


def ingest(docs: list[str], subject: str, batch_size: int, col: Collection, ef: BGEM3EmbeddingFunction):
    total = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        emb = ef(batch)
        sparse_matrix = CSRWithLen(emb["sparse"]).tocsr()
        entities = [
            batch,
            [subject] * len(batch),
            sparse_matrix,
            emb["dense"],
        ]
        col.insert(entities)
        total += len(batch)
        print(f"Inserted {total}/{len(docs)}")
    col.flush()
    col.load()
    print(f"Done. Collection entities: {col.num_entities}")


def main():
    # ===== 在此处直接填写参数，无需命令行 =====
    DATA_FILE = "./your_data.txt"           # 待导入的文本文件
    SEPARATOR = "\n---\n"                   # 文档分隔符，空块会被忽略
    SUBJECT = "general"                    # 可选标签字段
    BATCH_SIZE = 200                        # 每批插入数量
    URI = "http://localhost:19530"          # Milvus URI
    TOKEN = "root:Milvus"                  # Milvus token
    COLLECTION = "hybrid_search_collection" # 目标集合名
    PREVIEW = False                         # True 时仅预览前 10 条切分，不入库

    # Honor existing env settings for HF
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "True")

    if not os.getenv("HF_TOKEN"):
        print("[WARN] HF_TOKEN not set; set it via env or local_config if model needs authentication")

    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    dense_dim = ef.dim["dense"]

    col = ensure_collection(COLLECTION, dense_dim, URI, TOKEN)

    docs = load_docs(Path(DATA_FILE), SEPARATOR)
    if not docs:
        raise SystemExit("No documents found after splitting; check SEPARATOR or input file")

    print(f"Loaded {len(docs)} docs from {DATA_FILE}")

    if PREVIEW:
        preview_n = min(10, len(docs))
        print(f"Preview first {preview_n} docs (trimmed to 200 chars):")
        for idx, doc in enumerate(docs[:preview_n], start=1):
            snippet = doc if len(doc) <= 200 else doc[:200] + "..."
            print(f"[{idx}] {snippet}")
        return

    ingest(docs, SUBJECT, BATCH_SIZE, col, ef)


if __name__ == "__main__":
    main()
