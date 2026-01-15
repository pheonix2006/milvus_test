from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import uvicorn
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,host.docker.internal'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'

# 导入现有的检索逻辑
try:
    from milvus_test.hybrid_query_4partition import RouterRetriever, URI, TOKEN, COLLECTION_NAME
except ImportError:
    # 处理直接运行脚本时的路径问题
    from hybrid_query_4partition import RouterRetriever, URI, TOKEN, COLLECTION_NAME

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
    top_k: int = 3
    strategy: str = "global"

class SearchResponse(BaseModel):
    results: List[str]
    latency_ms: float

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
        
        # 3. 输出：返回包含文本片段的数组
        return SearchResponse(
            results=fragments,
            latency_ms=latency
        )
    except Exception as e:
        print(f"搜索出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 启动命令: python src/milvus_test/api.py
    # 或者使用 uvicorn: uvicorn src.milvus_test.api:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
