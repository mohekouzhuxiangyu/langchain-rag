"""命令行入口。

用法：
    python -m app.main init                 # 初始化数据库（建库 + pgvector + 建表）
    python -m app.main ingest [data.json]   # 入库（默认 data/samples.json）
    python -m app.main ask "问题" [--agent] # 单次问答（--agent 走 ReAct 多步智能体）
    python -m app.main chat  [--agent]      # 交互式对话
"""
from __future__ import annotations

import argparse
import json
import sys

from app.config import settings
from app.db import init_db
from app.ingest import load_items_from_json, upsert_media
from app.graph import build_agentic_rag, build_react_agent, run_agentic_rag
from app.retriever import hybrid_search, title_lookup


def _fmt_answer(result: dict) -> str:
    lines = [result["answer"].strip(), ""]
    if result.get("sources"):
        lines.append("— 来源 —")
        seen = set()
        for s in result["sources"]:
            key = (s.get("type"), s.get("media_id"), s.get("title"), s.get("url"))
            if key in seen:
                continue
            seen.add(key)
            if s.get("type") == "web":
                lines.append(f"  [联网] {s.get('title')} {s.get('url') or ''}")
            else:
                lines.append(
                    f"  [知识库]《{s.get('title')}》({s.get('type')}) 段落:{s.get('section')} 相关度:{s.get('score')}"
                )
    if result.get("trace"):
        lines.append("")
        lines.append("— 执行轨迹 —")
        lines.extend(f"  · {t}" for t in result["trace"])
    return "\n".join(lines)


def cmd_init(_args) -> None:
    init_db()
    print("数据库初始化完成。")


def cmd_ingest(args) -> None:
    items = load_items_from_json(args.data)
    print(f"加载 {len(items)} 条媒体数据...")
    stat = upsert_media(items)
    print(f"入库完成：媒体 {stat['media']} 条，向量块 {stat['chunks']} 个")


def cmd_ask(args) -> None:
    if args.agent:
        agent = build_react_agent()
        result = agent.invoke({"messages": [("user", args.query)]})
        answer = result["messages"][-1].content
        print(_fmt_answer({"answer": answer, "sources": [], "trace": []}))
        return
    print(_fmt_answer(run_agentic_rag(args.query)))


def cmd_chat(args) -> None:
    if args.agent:
        agent = build_react_agent()
        messages = []
        print("ReAct 智能体模式已开启（Ctrl+C 退出）")
        while True:
            try:
                q = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                break
            messages.append(("user", q))
            result = agent.invoke({"messages": messages})
            messages = result["messages"]
            print(f"\n助手: {messages[-1].content}")
        return

    graph = build_agentic_rag()
    print("Agentic RAG 已就绪（Ctrl+C 退出）")
    while True:
        try:
            q = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            break
        print(_fmt_answer(graph.invoke(
            {
                "query": q,
                "docs": [],
                "relevant": [],
                "needs_web": False,
                "web_context": "",
                "web_results": [],
                "answer": "",
                "sources": [],
                "trace": [],
            }
        )))


def cmd_retrieve(args) -> None:
    hits = hybrid_search(args.query, media_type=args.type or None, top_k=settings.top_k)
    for h in hits:
        print(f"[{h.score:.3f}] 《{h.title}》({h.type}/{h.section}) {h.content[:120]}...")
    if not hits:
        print("无结果")


def main() -> None:
    parser = argparse.ArgumentParser(prog="media-rag", description="影剧漫 Agentic RAG 知识库")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="初始化数据库")
    p_init.set_defaults(func=cmd_init)

    p_ingest = sub.add_parser("ingest", help="入库媒体数据")
    p_ingest.add_argument("data", nargs="?", default="data/samples.json")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="单次问答")
    p_ask.add_argument("query")
    p_ask.add_argument("--agent", action="store_true", help="使用 ReAct 多步智能体")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="交互式对话")
    p_chat.add_argument("--agent", action="store_true", help="使用 ReAct 多步智能体")
    p_chat.set_defaults(func=cmd_chat)

    p_ret = sub.add_parser("retrieve", help="仅检索测试")
    p_ret.add_argument("query")
    p_ret.add_argument("--type", choices=["movie", "tv", "anime"], default=None)
    p_ret.set_defaults(func=cmd_retrieve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
