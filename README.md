# Milvus 混合检索系统

基于 Milvus 向量数据库的混合语义检索系统，支持稠密向量与稀疏向量的联合检索，并提供 FastAPI 接口用于生产环境集成。

## 目录

- [项目简介](#项目简介)
- [技术背景](#技术背景)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用指南](#使用指南)
- [项目结构](#项目结构)
- [API 接口文档](#api-接口文档)
- [常见问题](#常见问题)

---

## 项目简介

本项目实现了一个完整的语义检索系统，核心功能包括：

- **混合检索**: 结合稠密向量（Dense）和稀疏向量（Sparse）的优势，提高检索准确率
- **分区路由**: 支持按业务领域分区存储和检索，实现查询意图路由
- **重排序优化**: 集成 BGE Reranker 模型对召回结果进行精排
- **RESTful API**: 提供 FastAPI 接口，便于集成到现有系统

### 为什么需要混合检索？

| 检索方式 | 优点 | 缺点 |
|---------|------|------|
| 关键词检索 | 精确匹配，适合专有名词 | 无法理解语义 |
| 稠密向量 | 理解语义相似性 | 可能遗漏精确关键词 |
| 稀疏向量 | 保留关键词权重 | 计算开销大 |
| **混合检索** | **兼顾语义与关键词** | **需要融合策略** |

本项目使用 **RRF (Reciprocal Rank Fusion)** 算法融合两种检索结果，取长补短。

---

## 技术背景

### 核心组件

1. **Milvus**: 开源向量数据库，用于存储和检索高维向量
2. **BGE-M3**: 嵌入模型，将文本转换为稠密+稀疏向量表示
3. **FastAPI**: Python Web 框架，提供 RESTful API 接口
4. **Xinference**: 模型服务框架，用于部署 Reranker 模型

### 混合检索流程

```
用户查询 "Python 装饰器怎么用"
    ↓
BGE-M3 向量化
    ↓
┌─────────────────────────────┐
│  稠密向量检索 (语义相似)      │
│  稀疏向量检索 (关键词匹配)    │
└─────────────────────────────┘
    ↓
RRF 融合排序
    ↓
Top-K 召回结果 (如 60 条)
    ↓
BGE Reranker 精排
    ↓
最终 Top-N 结果 (如 5 条)
```

---

## 环境准备

### 系统要求

- **操作系统**: Windows / Linux / macOS
- **Python**: >= 3.11
- **内存**: 建议 8GB+ (模型加载需要)
- **磁盘**: 建议 10GB+ 可用空间

### 依赖服务

#### 1. Milvus 数据库

推荐使用 Docker Compose 部署（更简单，包含所有依赖组件）：

**建议在单独的文件夹中运行 Milvus，避免与项目文件混在一起**：

```powershell
# Windows PowerShell
# 1. 创建一个独立的文件夹
mkdir C:\milvus
cd C:\milvus

# 2. 下载 docker-compose 配置文件
Invoke-WebRequest https://github.com/milvus-io/milvus/releases/download/v2.6.9/milvus-standalone-docker-compose.yml -OutFile docker-compose.yml

# 3. 启动 Milvus
docker compose up -d
```

```bash
# Linux/macOS
# 1. 创建一个独立的文件夹
mkdir -p ~/milvus
cd ~/milvus

# 2. 下载 docker-compose 配置文件
wget https://github.com/milvus-io/milvus/releases/download/v2.6.9/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 3. 启动 Milvus
docker compose up -d
```

启动后会看到：
```
Creating milvus-etcd        ... done
Creating milvus-minio      ... done
Creating milvus-standalone ... done
```

**验证安装**：
```powershell
# Windows PowerShell
curl http://localhost:19530/healthz
# 返回: OK
```

**常用命令**：
```powershell
# 查看运行状态
docker compose ps

# 停止 Milvus
docker compose down

# 重启 Milvus
docker compose restart
```

#### 2. Xinference (可选，用于 Rerank)

推荐使用 Docker 部署（避免 Python 环境冲突）：

**CPU 版本**（适用于没有 GPU 的环境）：

```powershell
# Windows PowerShell
# 1. 创建一个独立的文件夹用于存放 Xinference 数据
mkdir C:\xinference
cd C:\xinference

# 2. 启动 Xinference 容器 (CPU 版本)
docker run -d `
  --restart=always `
  --name=xinference `
  -v C:\xinference\data:/opt/xinference `
  -e XINFERENCE_HOME=/opt/xinference `
  -p 9997:9997 `
  docker-registry.neuedu.com/xprobe/xinference:v0.15.2-cpu `
  xinference-local -H 0.0.0.0
```

```bash
# Linux/macOS
# 1. 创建一个独立的文件夹
mkdir -p ~/xinference/data
cd ~/xinference

# 2. 启动 Xinference 容器 (CPU 版本)
docker run -d \
  --restart=always \
  --name=xinference \
  -v ~/xinference/data:/opt/xinference \
  -e XINFERENCE_HOME=/opt/xinference \
  -p 9997:9997 \
  docker-registry.neuedu.com/xprobe/xinference:v0.15.2-cpu \
  xinference-local -H 0.0.0.0
```

**验证安装**：
```powershell
curl http://localhost:9997/v1/models
# 返回可用的模型列表
```

**启动 Reranker 模型**：
```powershell
# 通过 API 启动模型
curl -X POST http://localhost:9997/v1/models \
  -H "Content-Type: application/json" \
  -d "{\"model_name\": \"bge-reranker-v2-m3\", \"model_type\": \"rerank\"}"
```

**常用命令**：
```powershell
# 查看容器状态
docker ps | findstr xinference

# 查看日志
docker logs xinference

# 停止容器
docker stop xinference

# 启动容器
docker start xinference

# 删除容器（谨慎）
docker rm -f xinference
```

---

**GPU 版本**（适用于有 NVIDIA GPU 的环境）：

GPU 版本需要安装 `nvidia-container-toolkit`，请参考 Xinference 官方文档获取镜像地址和部署步骤：

> **官方文档**: [Xinference Docker 部署指南](https://inference.readthedocs.io/en/latest/getting_started/installation.html#docker-image)

---

### 通过 Web 界面下载 Reranker 模型

Xinference 启动后，可以通过浏览器访问 Web 界面：

1. 打开浏览器访问: **http://localhost:9997**

2. 在界面中点击 **"Launch Model"** 启动模型

3. 选择模型类型和名称：
   - **Model Type**: 选择 `rerank`
   - **Model Name**: 输入或选择 `bge-reranker-v2-m3`

4. 点击 **"Launch"** 下载并启动模型

5. 模型下载完成后会自动启动，即可通过 API 调用

**Web 界面优势**：
- 可视化操作，无需记忆命令
- 支持浏览和搜索所有可用模型
- 可以管理多个模型的运行状态
- 查看模型资源使用情况

## 快速开始

### 1. 安装项目依赖

```bash
# 使用 uv (推荐)
pip install uv
uv sync

# 或使用传统 pip
pip install -r requirements.txt
```

### 2. 配置文件设置

```bash
# 复制配置模板
cp src/milvus_test/local_config.json.template src/milvus_test/local_config.json

# 编辑配置文件，填入你的 HF Token
# 获取 Token: https://huggingface.co/settings/tokens
```

### 3. 准备数据

将待检索的文档放入 `test_txt/` 目录，每个文件用 `# separator` 分隔文档块：

```text
# separator
这是第一段文档内容...

# separator
这是第二段文档内容...
```

### 4. 数据入库

```bash
uv run python -m milvus_test.unified_ingest
```

### 5. 启动 API 服务

```bash
uv run python -m milvus_test.api
# 服务将在 http://localhost:8000 启动
```

### 6. 测试检索

```bash
uv run python tests/test_api.py
```

---

## 配置说明

### 环境变量配置 (推荐)

项目支持通过 `.env` 文件管理配置，适用于生产环境和敏感信息管理：

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填写实际配置
```

**环境变量说明**：

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `AZURE_API_KEY` | Azure OpenAI API 密钥 | `your_api_key_here` |
| `AZURE_BASE_URL` | Azure API 完整地址 | `https://xxx.openai.azure.com/...?api-version=xxx` |
| `MILVUS_URI` | Milvus 服务地址 | `http://localhost:19530` |
| `MILVUS_TOKEN` | Milvus 认证令牌 | `root:Milvus` |
| `HF_ENDPOINT` | HuggingFace 镜像地址 | `https://hf-mirror.com` |

### 本地配置文件

`local_config.json` 用于数据集和检索策略配置：

```json
{
  "HF_TOKEN": "你的 Hugging Face Token",
  "active_dataset": "tech",
  "datasets": {
    "tech": {
      "collection_name": "hybrid_rag_collection_v1_test",
      "data_dir": "test_txt",
      "file_to_partition_map": {
        "ai.txt": "partition_ai",
        "cpp.txt": "partition_cpp",
        "python.txt": "partition_py",
        "machine_learning.txt": "partition_ml"
      },
      "strategy_map": {
        "ai": ["partition_ai"],
        "python": ["partition_py"],
        "global": []
      }
    }
  }
}
```

### 配置项说明

| 字段 | 说明 |
|------|------|
| `HF_TOKEN` | Hugging Face 访问令牌，用于下载 BGE-M3 模型 |
| `active_dataset` | 当前激活的数据集名称 |
| `collection_name` | Milvus 集合名称（类似数据库表名） |
| `data_dir` | 数据文件所在目录（相对于项目根目录的路径） |
| `file_to_partition_map` | 文件名到分区的映射关系 |
| `strategy_map` | 检索策略到分区列表的映射，空数组 `[]` 表示全局搜索 |

---

## 配置自己的文档

### 1. 准备文档文件

将你的文档放入指定目录（如 `test_txt/` 或自定义目录），每个文件用 `# separator` 分隔文档块：

```text
# separator
这是第一段文档内容。可以是一篇文章、一个知识条目或任意文本片段。

# separator
这是第二段文档内容。每段会被当作独立的检索单元。
```

**文件格式要求**：
- 支持 `.txt`、`.md` 等纯文本格式
- 编码格式：UTF-8
- 分隔符：必须是 `# separator`（前后无空格）

### 2. 修改 local_config.json

根据你的文档结构修改配置：

```json
{
  "HF_TOKEN": "你的 Hugging Face Token",
  "active_dataset": "my_data",
  "datasets": {
    "my_data": {
      "collection_name": "my_collection",
      "data_dir": "my_docs",
      "file_to_partition_map": {
        "产品手册.txt": "partition_product",
        "技术文档.txt": "partition_tech",
        "FAQ.txt": "partition_faq"
      },
      "strategy_map": {
        "product": ["partition_product"],
        "tech": ["partition_tech"],
        "faq": ["partition_faq"],
        "global": []
      }
    }
  }
}
```

### 3. 参数调整说明

| 参数 | 调整建议 |
|------|----------|
| `collection_name` | 更换为新名称，避免与旧数据冲突 |
| `data_dir` | 指向你存放文档的文件夹路径 |
| `file_to_partition_map` | 左边是文件名，右边是分区名（自定义，用英文/下划线） |
| `strategy_map` | 定义检索路由，键名是策略名，值是该策略要搜索的分区列表 |

**分区命名规则**：
- 只能包含字母、数字、下划线
- 不能以数字开头
- 建议使用 `partition_` 前缀便于识别

**检索策略说明**：
- `"global": []` - 空数组表示全局搜索，检索所有分区
- `["partition_xxx"]` - 只在指定分区中检索
- 一个策略可以对应多个分区：`["partition_a", "partition_b"]`

---

## 使用指南

### 命令行检索

```bash
uv run python -m milvus_test.unified_query
```

### Python 代码调用

```python
from milvus_test.unified_query import RouterRetriever, URI, TOKEN

# 初始化检索器
retriever = RouterRetriever(URI, TOKEN)

# 执行检索
results, latency = retriever.search(
    query_text="Python 装饰器的作用是什么",
    intent="python",  # 或 "global" 进行全局搜索
    top_k=5
)

# 查看结果
for hit in results:
    print(f"Score: {hit.score:.4f}")
    print(f"Content: {hit.entity.get('content')}")
```

---

## 项目结构

```
milvus_test/
├── src/milvus_test/
│   ├── __init__.py
│   ├── api.py                      # FastAPI 服务接口 (含 /search, /rerank, /chat)
│   ├── unified_query.py            # 统一检索逻辑
│   ├── unified_ingest.py           # 统一入库逻辑
│   ├── local_config.json           # 本地配置（含敏感信息，不提交）
│   └── local_config.json.template  # 配置模板
├── tests/
│   ├── test_api.py                 # API 接口测试
│   ├── test_api_chat.py            # Chat API 测试
│   ├── test_custom_azure_api.py    # Azure API 直接测试
│   ├── test_full_search_rerank.py  # 完整流程测试
│   ├── test_main.py                # 基础功能测试
│   ├── 全量测试milvus.yml          # 测试配置
│   └── notebooks/                  # Jupyter Notebook 测试文件
│       ├── hybrid_query_4partition_test.ipynb
│       └── rerank_debug_test.ipynb
├── scripts/
│   └── get_models.py               # 辅助脚本
├── test_txt/                       # 测试数据目录
├── .env.example                    # 环境变量配置模板
├── pyproject.toml                  # 项目依赖配置
└── README.md                       # 本文档
```

---

## API 接口文档

### 1. 混合检索接口 `/search`

**请求示例**:

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是机器学习？",
    "top_k": 60,
    "strategy": "ml"
  }'
```

**响应示例**:

```json
{
  "pure_documents": [
    "机器学习是一门多领域交叉学科...",
    "深度学习是机器学习的一个子集..."
  ]
}
```

### 2. 重排序接口 `/rerank`

**请求示例**:

```bash
curl -X POST "http://localhost:8000/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是机器学习？",
    "documents": {"pure_documents": ["文档1...", "文档2..."]},
    "top_k": 3,
    "score_threshold": 0.5
  }'
```

**响应示例**:

```json
{
  "pure_documents": ["文档1..."],
  "formatted_result": "资料来源 [1]\n文档1..."
}
```

### 3. 聊天接口 `/chat` (Azure OpenAI)

**请求示例**:

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请用一句话介绍一下什么是向量数据库。",
    "model": "gpt-5.1",
    "system_prompt": "你是一个严谨的计算机科学家。"
  }'
```

**响应示例**:

```json
{
  "answer": "向量数据库是一种专门用于存储、索引和查询高维向量数据的数据库系统..."
}
```

**说明**：
- 需要在 `.env` 文件中配置 `AZURE_API_KEY` 和 `AZURE_BASE_URL`
- 支持自定义 Azure 定制格式的 API（`input`/`output` 字段结构）

### 完整调用流程

```python
import requests

BASE_URL = "http://localhost:8000"

def search_with_rerank(query: str, strategy: str = "global", top_k: int = 5):
    # 1. 混合检索召回
    search_res = requests.post(f"{BASE_URL}/search", json={
        "query": query,
        "top_k": 60,
        "strategy": strategy
    }).json()

    # 2. Rerank 精排
    rerank_res = requests.post(f"{BASE_URL}/rerank", json={
        "query": query,
        "documents": search_res,
        "top_k": top_k
    }).json()

    return rerank_res["formatted_result"]

# 使用
context = search_with_rerank("Python 装饰器的作用", "python")
print(context)
```

---

## 常见问题

### Q1: 如何获取 Hugging Face Token?

1. 访问 https://huggingface.co/
2. 注册/登录账号
3. 进入 Settings → Access Tokens
4. 创建新 Token 并复制

### Q2: 首次运行很慢怎么办?

首次运行需要下载 BGE-M3 模型（约 2GB），下载后会自动缓存到本地。后续启动会快很多。

可以设置国内镜像加速：
```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### Q3: Milvus 连接失败

检查 Docker 容器是否运行：
```bash
docker ps | grep milvus
```

检查端口是否被占用：
```bash
netstat -an | findstr 19530  # Windows
netstat -an | grep 19530     # Linux/Mac
```

### Q4: 如何添加新的数据集?

在 `local_config.json` 的 `datasets` 中添加新配置：

```json
{
  "datasets": {
    "my_new_dataset": {
      "collection_name": "my_collection",
      "data_dir": "my_data_dir",
      "file_to_partition_map": {
        "file1.txt": "partition_1"
      },
      "strategy_map": {
        "category_a": ["partition_1"],
        "global": []
      }
    }
  },
  "active_dataset": "my_new_dataset"
}
```

### Q5: Rerank 接口报错

确保 Xinference 服务正在运行：
```bash
curl http://localhost:9997/v1/models
```

确保 Reranker 模型已启动：
```bash
xinference list -u bge-reranker-v2-m3
```

---

## 许可证

本项目仅供内部使用。

## 联系方式

如有问题请联系开发团队。
