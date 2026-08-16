"""LangGraph Agentic RAG：路由 -> 检索 -> 相关性评分 -> 生成 / 联网兜底。"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.config import settings
from app.models import MEDIA_TYPE_LABEL, SearchHit
from app.retriever import hybrid_search
from app.tools import MEDIA_TOOLS, web_search_duckduckgo


def get_llm(temperature: float = 0.2):
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
        max_tokens=2048,
    )


# ---------------- State ----------------

class RAGState(TypedDict):
    query: str
    intent: str                     # media | general
    media_type: Optional[str]       # movie | tv | anime | None
    keywords: str
    docs: list[SearchHit]
    relevant: list[SearchHit]
    needs_web: bool
    web_context: str
    web_results: list[dict]
    answer: str
    sources: list[dict]
    trace: list[str]


# ---------------- 结构化输出 Schema ----------------

class RouteDecision(BaseModel):
    intent: Literal["media", "general"] = Field(description="media=与电影/电视剧/动漫资源相关；general=其他")
    media_type: Optional[Literal["movie", "tv", "anime"]] = Field(
        default=None, description="用户明确指定类型时填 movie/tv/anime，否则 null"
    )
    keywords: str = Field(description="用于检索的规范化关键词（保留片名、人名、类型词）")


class GradeDecision(BaseModel):
    relevant_indices: list[int] = Field(description="与问题相关的结果序号列表（从 0 开始）")


# ---------------- 资源查询判定 ----------------

RESOURCE_HINTS = ["观看", "资源", "下载", "链接", "在线", "在哪", "哪里", "网址", "地址", "播放", "平台", "免费", "怎么看"]


def is_resource_query(query: str) -> bool:
    """判断问题是否偏向「找资源/观看/下载链接」。"""
    return any(h in query for h in RESOURCE_HINTS)


# ---------------- Nodes ----------------

def _call_structured(llm, decision_cls: type[BaseModel], prompt: ChatPromptTemplate, **kw):
    """bind_tools + 手动解析（auto 模式，含 JSON 内容兜底），规避 schema 兼容问题。"""
    import json
    import re

    bound = llm.bind_tools([decision_cls], tool_choice="auto")
    ai = bound.invoke(prompt.format_messages(**kw))
    for tc in getattr(ai, "tool_calls", []) or []:
        if tc.get("name") == decision_cls.__name__:
            return decision_cls(**tc["args"])
    # 兜底：模型未走工具调用时，尝试从内容中解析 JSON
    content = (ai.content or "").strip()
    m = re.search(r"\{.*\}", content, re.S)
    if m:
        try:
            return decision_cls(**json.loads(m.group(0)))
        except Exception:  # noqa: BLE001
            pass
    raise RuntimeError(f"LLM 未返回结构化决策: {content[:200]}")


def route_node(state: RAGState) -> dict:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是媒体查询路由。判断用户问题是否关于电影/电视剧/动漫（含找资源、剧情、演员、评分、推荐）。"
                "media_type 仅在用户明确指定电影/电视剧/动漫时填写。",
            ),
            ("human", "用户问题：{query}"),
        ]
    )
    dec = _call_structured(llm, RouteDecision, prompt, query=state["query"])
    return {
        "intent": dec.intent,
        "media_type": dec.media_type,
        "keywords": dec.keywords or state["query"],
        "trace": [f"路由: intent={dec.intent}, media_type={dec.media_type or '未指定'}"],
    }


def retrieve_node(state: RAGState) -> dict:
    hits = hybrid_search(state["keywords"], media_type=state.get("media_type"), top_k=settings.top_k)
    return {
        "docs": hits,
        "trace": state.get("trace", []) + [f"检索: 命中 {len(hits)} 条（top_k={settings.top_k}）"],
    }


def grade_node(state: RAGState) -> dict:
    docs = state.get("docs", [])
    if not docs:
        return {"relevant": [], "needs_web": True, "trace": state.get("trace", []) + ["评分: 无检索结果"]}
    llm = get_llm(temperature=0)
    numbered = "\n\n".join(
        f"[{i}] 《{d.title}》({d.type}, {d.section}) 相关度 {d.score:.3f}\n{d.content}" for i, d in enumerate(docs)
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是检索相关性评审。根据用户问题判断哪些检索结果真正相关（能直接帮助回答），"
                "返回相关结果的序号列表；都不相关则返回空列表 []。",
            ),
            ("human", "用户问题：{query}\n\n检索结果：\n{numbered}"),
        ]
    )
    try:
        dec = _call_structured(llm, GradeDecision, prompt, query=state["query"], numbered=numbered)
    except Exception as e:  # noqa: BLE001
        print(f"[grade] 解析失败，视为全部相关: {e}")
        dec = GradeDecision(relevant_indices=list(range(len(docs))))
    relevant = [docs[i] for i in dec.relevant_indices if 0 <= i < len(docs)]
    needs_web = len(relevant) == 0
    return {
        "relevant": relevant,
        "needs_web": needs_web,
        "trace": state.get("trace", []) + [f"评分: {len(relevant)} 条相关, needs_web={needs_web}"],
    }


def web_search_node(state: RAGState) -> dict:
    q = state["keywords"]
    if state.get("media_type"):
        q = f"{MEDIA_TYPE_LABEL[state['media_type']]} {q}"
    q += " 资源 在线观看"
    results = web_search_duckduckgo(q, max_results=8)
    if not results:
        results = web_search_duckduckgo(state["keywords"], max_results=8)
    ctx = "\n\n".join(
        f"[{i+1}] {r['title']}\n   URL: {r['url']}\n   {r['snippet']}" for i, r in enumerate(results)
    )
    return {
        "web_context": ctx,
        "web_results": results,
        "trace": state.get("trace", []) + [f"联网搜索: {len(results)} 条结果"],
    }


def resource_search_node(state: RAGState) -> dict:
    """资源类问题：知识库有相关内容但用户要「观看/下载链接」时，联网补充真实链接。"""
    title = state.get("keywords") or state.get("query", "")
    results = web_search_duckduckgo(f"{title} 在线观看", max_results=8)
    if not results:
        results = web_search_duckduckgo(f"{title} 资源 下载", max_results=8)
    if not results and state.get("media_type"):
        results = web_search_duckduckgo(f"{MEDIA_TYPE_LABEL[state['media_type']]} {title} 免费观看", max_results=8)
    ctx = "\n\n".join(
        f"[{i+1}] {r['title']}\n   URL: {r['url']}\n   {r['snippet']}" for i, r in enumerate(results)
    )
    return {
        "web_context": ctx,
        "web_results": results,
        "trace": state.get("trace", []) + [f"资源链接补充: 联网返回 {len(results)} 条"],
    }


def generate_node(state: RAGState) -> dict:
    llm = get_llm()
    relevant = state.get("relevant", [])
    # 本地知识库上下文（带编号，供引用）
    ctx_parts = []
    sources: list[dict] = []
    for i, d in enumerate(relevant, 1):
        label = MEDIA_TYPE_LABEL.get(d.type, d.type)
        ctx_parts.append(f"[{i}] 《{d.title}》({label}) 段落: {d.section}\n{d.content}")
        sources.append(d.to_source())
    local_ctx = "\n\n".join(ctx_parts) if ctx_parts else "（无）"
    web_ctx = state.get("web_context") or "（无）"

    for r in state.get("web_results") or []:
        sources.append(
            {
                "type": "web",
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", ""),
            }
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是「全网影剧漫资源助手」，基于本地知识库（PostgreSQL/pgvector）检索结果与联网信息回答用户关于"
                "电影/电视剧/动漫的问题（剧情、演员、评分、推荐、资源与观看途径等）。\n"
                "要求：\n"
                "1. 优先用本地知识库内容作答；知识库不足时再结合联网信息，并注明\"（联网信息）\"。\n"
                "2. 引用来源：正文用 [1][2] 标注，引用编号对应下方提供的编号内容。\n"
                "3. 资源/观看/下载类问题：必须把联网结果中的**具体链接**以 https://... 原样列出（供用户直接点击），"
                "格式如「平台名: https://...」，并标注 [联网]；若没有真实链接则如实说明，绝不编造链接。\n"
                "4. 使用简体中文，条理清晰。",
            ),
            (
                "human",
                "用户问题：{query}\n\n"
                "【本地知识库检索结果】\n{local_ctx}\n\n"
                "【联网搜索结果】\n{web_ctx}\n\n"
                "请回答（正文中标注 [编号] 引用）。",
            ),
        ]
    )
    answer = llm.invoke(
        prompt.format_messages(query=state["query"], local_ctx=local_ctx, web_ctx=web_ctx)
    ).content

    return {
        "answer": answer,
        "sources": sources,
        "trace": state.get("trace", []) + ["生成: 完成"],
    }


def general_node(state: RAGState) -> dict:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是「全网影剧漫资源助手」。用户问题与电影/电视剧/动漫无关时，直接给出简洁友好的回答。",
            ),
            ("human", "{query}"),
        ]
    )
    answer = llm.invoke(prompt.format_messages(query=state["query"])).content
    return {
        "answer": answer,
        "sources": [],
        "trace": state.get("trace", []) + ["普通问答（未走检索）"],
    }


def finalize_node(state: RAGState) -> dict:
    trace = state.get("trace", [])
    return {"trace": trace + ["完成"]}


# ---------------- Graph ----------------

def build_agentic_rag() -> object:
    g = StateGraph(RAGState)

    g.add_node("route", route_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("web_search", web_search_node)
    g.add_node("resource_search", resource_search_node)
    g.add_node("generate", generate_node)
    g.add_node("general", general_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route",
        lambda s: "retrieve" if s.get("intent") == "media" else "general",
        {"retrieve": "retrieve", "general": "general"},
    )
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        lambda s: (
            "web_search"
            if s.get("needs_web")
            else ("resource_search" if is_resource_query(s.get("query", "")) else "generate")
        ),
        {
            "web_search": "web_search",
            "resource_search": "resource_search",
            "generate": "generate",
        },
    )
    g.add_edge("web_search", "generate")
    g.add_edge("resource_search", "generate")
    g.add_edge("generate", "finalize")
    g.add_edge("general", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


def build_react_agent() -> object:
    """ReAct 多步智能体：可自主调用 知识库检索 / 元数据查询 / 联网搜索 三个工具。"""
    from langgraph.prebuilt import create_react_agent

    llm = get_llm(temperature=0.3)
    system = (
        "你是「全网影剧漫资源助手」。回答电影/电视剧/动漫相关问题（剧情、演员、评分、推荐、资源与观看途径）。\n"
        "工具使用策略：\n"
        "1. 先调用 search_media_kb 查本地知识库；片名已知时可用 lookup_media_meta 查结构化元数据。\n"
        "2. 知识库不足、或需要资源/观看/下载渠道时，调用 search_web 联网补充，并把具体链接 https://... 原样列在回答中供用户点击。\n"
        "3. 最终回答用简体中文，标注来源（本地库/联网）。"
    )
    return create_react_agent(llm, MEDIA_TOOLS, prompt=system)


def run_agentic_rag(query: str) -> dict:
    graph = build_agentic_rag()
    out = graph.invoke(
        {
            "query": query,
            "docs": [],
            "relevant": [],
            "needs_web": False,
            "web_context": "",
            "web_results": [],
            "answer": "",
            "sources": [],
            "trace": [],
        }
    )
    return {"answer": out["answer"], "sources": out["sources"], "trace": out["trace"]}
