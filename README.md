# 影剧漫 Agentic RAG 知识库系统

基于 **LangChain + LangGraph** 构建的智能体 RAG 知识库，使用 **PostgreSQL + pgvector** 存储向量化数据，
LLM 选用 **DeepSeek**，面向**全网电影 / 电视剧 / 动漫资源查询**（剧情、演员、评分、推荐、资源与观看途径）。

## 架构

```
用户问题
   │
   ▼
┌─────────┐   路由（LLM 结构化决策：是否媒体类？类型？检索关键词）
│  route  │──────────────────────────┐
└────┬────┘                          │ general（非媒体问题）
     │ media                         ▼
     ▼                          ┌──────────┐
┌──────────┐  混合检索          │ general  │  直接 LLM 回答
│ retrieve │  pgvector 向量相似度          └────┬─────┘
└────┬─────┘  + PostgreSQL 全文检索              │
     │        + 标题/别名精确匹配（RRF 融合）       │
     ▼                                          │
┌──────────┐  相关性评分（LLM 评审）               │
│  grade   │──────────────────────┐             │
└────┬─────┘                      │ 无相关结果    │
     │ 有相关结果                   ▼             │
     ▼                       ┌────────────┐      │
┌──────────┐                │ web_search │      │
│ generate │◄───────────────│ 联网兜底    │      │
│ 带引用回答 │                └────────────┘      │
└────┬─────┘                                     │
     ▼                                           ▼
┌──────────┐                                ┌──────────┐
│ finalize │                                │ finalize │
└──────────┘                                └──────────┘
```

- **自定义 Agentic RAG 图**（`app/graph.py`）：route → retrieve → grade → generate / web_search 的条件流。
- **ReAct 多步智能体**（`build_react_agent`）：可自主调用「知识库检索 / 元数据查询 / 联网搜索」三个工具，适合多跳复杂问题。
- **混合检索**（`app/retriever.py`）：pgvector 向量相似度 + PostgreSQL `to_tsvector` 全文检索 + 标题/别名精确匹配，RRF 分数融合。
- **嵌入模型**：本地 `BAAI/bge-small-zh-v1.5`（中文，512 维，经 HF 镜像下载），无需额外 API key。
- **联网兜底**：DuckDuckGo 搜索（免 key），知识库无相关内容时自动触发。

## 目录结构

```
langchain-rag/
├── app/
│   ├── config.py      # 配置（.env）
│   ├── db.py          # 建库 / pgvector / 建表
│   ├── models.py      # 媒体数据模型
│   ├── embeddings.py  # 本地中文嵌入模型
│   ├── ingest.py      # 分块 + 向量化 + upsert
│   ├── retriever.py   # 混合检索（向量+FTS+标题，RRF）
│   ├── tools.py       # 知识库检索 / 元数据 / 联网 工具
│   ├── graph.py       # LangGraph Agentic RAG + ReAct 智能体
│   └── main.py        # CLI 入口
├── data/samples.json  # 种子数据（73 条：电影25/剧集23/动漫25）
├── scripts/setup_db.sh
└── tests/test_e2e.py  # 端到端验证
```

## 环境要求

- macOS（本仓库在 arm64 Mac 验证）或 Linux
- Homebrew（macOS 安装 PostgreSQL 用），Python 3.12

> **注意**：Homebrew 的 `pgvector` 预编译包只支持 `postgresql@17`/`@18`（不含 16），
> 因此本项目统一使用 **postgresql@17**。若换用其他 PG 大版本，需要对应版本的 pgvector
> （`scripts/setup_db.sh` 会在缺失时尝试源码编译）。

## 快速开始

```bash
# 1) 安装依赖（Python 3.12 + PostgreSQL 17 + pgvector）
brew install python@3.12 postgresql@17 pgvector

# 2) 创建虚拟环境并安装 Python 依赖
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3) 初始化数据库（建库 + 启用 pgvector + 建表）
bash scripts/setup_db.sh          # 或手动：brew services start postgresql@17
python -m app.main init

# 4) 入库种子数据
python -m app.main ingest data/samples.json

# 5) 问答
python -m app.main ask "推荐几部高分科幻电影"
python -m app.main ask "《狂飙》高启强是什么人？" --agent   # ReAct 多步智能体
python -m app.main chat                                      # 交互式
```

## 配置（.env）

| 变量 | 说明 | 默认 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API key | 必填 |
| `DEEPSEEK_BASE_URL` | DeepSeek 接口地址 | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 模型 | `deepseek-chat` |
| `PG_HOST/PORT/USER/PASSWORD/DATABASE` | PostgreSQL 连接 | `127.0.0.1/5432/postgres/postgres/media_rag` |
| `HF_ENDPOINT` | HuggingFace 镜像（国内） | `https://hf-mirror.com` |
| `EMBEDDING_MODEL` | 嵌入模型 | `BAAI/bge-small-zh-v1.5` |
| `TOP_K` / `SCORE_THRESHOLD` | 检索参数 | `6` / `0.35` |

## 入库自己的数据

`data/samples.json` 是种子数据。接入真实数据源（如 TMDB/豆瓣/资源站爬虫）时，
把爬到的数据整理成同结构的 JSON（字段见 `app/models.py` 的 `MediaItem`），再执行：

```bash
python -m app.main ingest your_data.json   # 幂等：重复执行会重建对应条目向量
```

## 验证

```bash
python tests/test_e2e.py
```

覆盖：建库 → 入库 → 混合检索 → Agentic RAG 问答 → ReAct 智能体工具调用 → 联网兜底。

## 启动 Web 服务

内置 FastAPI Web 服务（网页聊天界面 + HTTP API）：

```bash
bash scripts/start_server.sh          # 默认端口 8080
# 或指定端口
MEDIA_RAG_PORT=9000 bash scripts/start_server.sh
```

启动后浏览器打开：http://127.0.0.1:8080

HTTP API：
- `GET /api/health` — 健康检查（含库内媒体/向量块数量）
- `POST /api/ask` — 问答，`{"query": "...", "agent": false}`（`agent: true` 走 ReAct 多步智能体）
- `GET /api/retrieve?query=...&media_type=movie` — 仅检索调试

## 示例

```
你: 推荐几部评分高的科幻电影
答: 结合知识库，推荐以下高分科幻电影：
  1. 《星际穿越》(9.4) - 诺兰执导，硬科幻+亲情 [1]
  2. 《盗梦空间》(9.3) - 多层梦境高概念 [2]
  3. 《流浪地球2》(8.3) - 国产硬科幻里程碑 [3]
  ...
```

## 说明

- 嵌入模型首次运行会从 HF 镜像下载（约 100MB），之后走本地缓存。
- 联网搜索走 DuckDuckGo（免 key），网络受限时自动降级为「仅知识库回答」。
- DeepSeek 不提供 embedding API，故嵌入用本地开源模型，LLM 全部走 DeepSeek。
