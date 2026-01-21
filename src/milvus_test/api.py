from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Union
import os
import uvicorn
import json
import requests

os.environ['NO_PROXY'] = 'localhost,127.0.0.1,host.docker.internal'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'

# 导入现有的检索逻辑
try:
    from milvus_test.hybrid_query_4partition_v2 import RouterRetriever, URI, TOKEN, COLLECTION_NAME
except ImportError:
    # 处理直接运行脚本时的路径问题
    from hybrid_query_4partition_v2 import RouterRetriever, URI, TOKEN, COLLECTION_NAME

app = FastAPI(
    title="Milvus Hybrid Search API",
    description="使用 FastAPI 包装的 Milvus 混合检索接口"
)

# 在应用启动时初始化检索器以避免重复加载模型
# 注意：在生产环境中，建议使用 lifespan 事件处理器
retriever = None

@app.on_event("startup")
def startup_event():
    global retriever
    print("正在初始化检索器...")
    retriever = RouterRetriever(URI, TOKEN, COLLECTION_NAME)
    print("检索器初始化完成。")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 60
    strategy: str = "global"

class SearchResponse(BaseModel):
    pure_documents: List[str]

class RerankRequest(BaseModel):
    model: str = "bge-reranker-v2-m3"
    query: str
    documents: Any  # 支持字符串 JSON 或直接的对象 {"pure_documents": [...]}
    top_k: int = 10
    score_threshold: float = -10.0

class RerankResponse(BaseModel):
    pure_documents: List[str]
    formatted_result: str

def main(input_data: Any) -> dict:
    """
    处理输入并提取结果列表。
    
    Args:
        input_data (Any): 序列化的 JSON 字符串或字典对象
    """
    try:
        # 如果是字符串，先解析
        if isinstance(input_data, str):
            body_data = json.loads(input_data)
        else:
            body_data = input_data

        if not isinstance(body_data, dict):
            return {"pure_documents": []}
        
        # 处理嵌套的 'body' 键 (有些来源会将内容再次序列化进 body 字段)
        if "body" in body_data and isinstance(body_data["body"], str):
            try:
                body_data = json.loads(body_data["body"])
            except json.JSONDecodeError:
                pass

        # 提取 results 列表，并确保其为 Array[String] 格式
        pure_documents = body_data.get('pure_documents') or body_data.get('results') or []
        
        # 严谨性校验：确保 pure_documents 确实是列表
        if not isinstance(pure_documents, list):
            pure_documents = []
            
    except (json.JSONDecodeError, TypeError):
        # 如果 body_str 不是合法的 JSON 格式，返回空列表
        pure_documents = []

    return {
        "pure_documents": pure_documents
    }

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest):
    """
    混合检索接口
    - **query**: 查询文本
    - **top_k**: 返回的结果数量 (默认 60)
    - **strategy**: 检索策略 (ai, cpp, python, ml, global)
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    try:
        # 1. 内部逻辑：向量化并在 Milvus 中检索
        hits, latency = retriever.search(
            query_text=req.query,
            intent=req.strategy,
            top_k=req.top_k
        )
        
        # 2. 提取文本片段
        fragments = [hit.entity.get("content") for hit in hits]
        
        # 3. 输出：返回封装好的纯文本列表
        return SearchResponse(
            pure_documents=fragments
        )
    except Exception as e:
        print(f"搜索出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rerank", response_model=RerankResponse)
async def rerank_endpoint(req: RerankRequest):
    """
    Rerank 重排序接口
    - **model**: 模型名称 (默认 bge-reranker-v2-m3)
    - **query**: 查询问题
    - **documents**: JSON 字符串格式的文档列表
    - **top_k**: 最终返回的文档数量
    - **score_threshold**: 得分阈值
    """
    try:
        # 1. 模仿 main 函数的逻辑提取纯文档列表
        # 如果 documents 本身就是一个包含 {"pure_documents": [...]} 的 JSON 字符串
        docs_data = main(req.documents)
        pure_docs = docs_data.get("pure_documents", [])
        
        if not pure_docs:
            return RerankResponse(pure_documents=[], formatted_result="")

        # 2. 调用 Xinference Rerank 服务 (分批处理以节省内存)
        base_url = "http://localhost:9997/v1"
        rerank_url = f"{base_url}/rerank"
        batch_size = 2
        all_scored_results = []
        
        print(f"正在发起分批重排序: '{req.query}' (总文档数: {len(pure_docs)}, 每批: {batch_size})")
        
        for i in range(0, len(pure_docs), batch_size):
            batch_docs = pure_docs[i : i + batch_size]
            payload = {
                "model": req.model,
                "query": req.query,
                "documents": batch_docs
            }
            
            # 增加单次 batch 的日志
            print(f"  正在处理第 {i//batch_size + 1} 批...")
            response = requests.post(rerank_url, json=payload, timeout=500)
            response.raise_for_status()
            batch_result = response.json()
            
            if "results" in batch_result:
                for item in batch_result["results"]:
                    # Xinference 返回得分字段通常为 "relevance_score"
                    score = item.get("relevance_score", 0)
                    doc_idx_in_batch = item["index"]
                    doc_text = batch_docs[doc_idx_in_batch]
                    all_scored_results.append((score, doc_text))

        # 3. 排序、阈值过滤与 top_k 截断
        all_scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # 过滤低于阈值的文档
        filtered_results = [item for item in all_scored_results if item[0] >= req.score_threshold]
        
        # 取 top_k
        final_results = filtered_results[:req.top_k]
        sorted_docs = [item[1] for item in final_results]
        
        # 4. 格式化为完整字符串
        formatted_parts = []
        for idx, doc in enumerate(sorted_docs, 1):
            formatted_parts.append(f"资料来源 [{idx}]\n{doc}")
        
        formatted_result = "\n\n".join(formatted_parts)
        
        return RerankResponse(
            pure_documents=sorted_docs,
            formatted_result=formatted_result
        )
        
    except Exception as e:
        print(f"重排序出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 启动命令: python src/milvus_test/api.py
    # 或者使用 uvicorn: uvicorn src.milvus_test.api:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
