"""Agent 工具集：媒体知识库检索 + 全网搜索（DuckDuckGo，免 key）。"""
from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool

from app.config import settings
from app.retriever import hybrid_search, title_lookup


def web_search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo 网页搜索（免 API key）。"""
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None

    if DDGS is not None:
        try:
            out = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    out.append(
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href", "") or r.get("url", ""),
                            "snippet": r.get("body", "") or r.get("snippet", ""),
                        }
                    )
            if out:
                return out
        except Exception as e:  # noqa: BLE001
            print(f"[web] ddgs 失败: {e}, 尝试 HTML 兜底")

    # 兜底：直接请求 DuckDuckGo html 版
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            timeout=15,
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for result in soup.select(".result")[:max_results]:
            a = result.select_one(".result__a")
            sn = result.select_one(".result__snippet")
            if a:
                results.append(
                    {
                        "title": a.get_text(strip=True),
                        "url": a.get("href", ""),
                        "snippet": sn.get_text(strip=True) if sn else "",
                    }
                )
        if results:
            return results
    except Exception as e:  # noqa: BLE001
        print(f"[web] HTML 兜底失败: {e}")
    return []


def _fmt_web(results: list[dict], max_len: int = 600) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r['title']}\n   URL: {r['url']}\n   {r['snippet'][:max_len]}")
    return "\n\n".join(parts)


@tool
def search_media_kb(query: str, media_type: Optional[str] = None) -> str:
    """从本地媒体知识库（PostgreSQL/pgvector，收录电影/电视剧/动漫的剧情、主创、评分、播放平台与资源途径）检索相关信息。
    参数 query 为检索词（可含片名），media_type 可选 movie/tv/anime。"""
    hits = hybrid_search(query, media_type=media_type, top_k=6)
    if not hits:
        return "知识库中未检索到相关内容。"
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] 《{h.title}》({h.type}) [{h.section}] 相关度 {h.score:.3f}\n{h.content}")
    return "\n\n".join(lines)


@tool
def lookup_media_meta(title: str) -> str:
    """按片名精确查媒体条目元数据（含别名、资源途径、平台等结构化字段）。"""
    rows = title_lookup(title)
    if not rows:
        return f"未找到《{title}》的条目。"
    out = []
    for r in rows[:3]:
        out.append(
            json.dumps(
                {
                    "id": r["id"],
                    "title": r["title"],
                    "aliases": r["aliases"],
                    "type": r["type"],
                    "year": r["year"],
                    "genres": r["genres"],
                    "rating": r["rating"],
                    "director": r["director"],
                    "cast": r["cast"],
                    "episodes": r["episodes"],
                    "status": r["status"],
                    "platform": r["platform"],
                    "resource": r["resource"],
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(out)


@tool
def search_web(query: str) -> str:
    """联网搜索全网公开信息（当前经 DuckDuckGo）。当知识库无法回答或需要最新资源/观看渠道时使用。"""
    results = web_search_duckduckgo(query, max_results=5)
    if not results:
        return "联网搜索未返回结果。"
    return _fmt_web(results)


MEDIA_TOOLS = [search_media_kb, lookup_media_meta, search_web]
