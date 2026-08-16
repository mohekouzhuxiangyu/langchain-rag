"""端到端验证：入库 -> 检索 -> Agentic RAG 问答 -> ReAct 智能体。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import init_db  # noqa: E402
from app.graph import build_agentic_rag, build_react_agent  # noqa: E402
from app.ingest import load_items_from_json, upsert_media  # noqa: E402
from app.retriever import hybrid_search, title_lookup  # noqa: E402


def check(ok: bool, msg: str) -> bool:
    print(("  [PASS] " if ok else "  [FAIL] ") + msg)
    return ok


def main() -> int:
    all_ok = True

    print("\n=== 1. 数据库初始化 ===")
    init_db()
    all_ok &= check(True, "pgvector 扩展与表结构就绪")

    print("\n=== 2. 数据入库（幂等） ===")
    items = load_items_from_json("data/samples.json")
    stat = upsert_media(items)
    all_ok &= check(stat["media"] >= 50, f"入库媒体 {stat['media']} 条（期望>=50）")
    all_ok &= check(stat["chunks"] > 100, f"向量块 {stat['chunks']} 个（期望>100）")

    print("\n=== 3. 混合检索验证 ===")
    q = "流浪地球"
    hits = hybrid_search(q, top_k=5)
    all_ok &= check(len(hits) > 0, f"检索《{q}》返回 {len(hits)} 条")
    if hits:
        print(f"      top1: {hits[0].title} score={hits[0].score:.3f}")

    hits2 = hybrid_search("张译 主演 电视剧", media_type="tv", top_k=5)
    all_ok &= check(len(hits2) > 0, "类型过滤检索 tv 返回结果")
    if hits2:
        print(f"      top1: {hits2[0].title} score={hits2[0].score:.3f}")

    tl = title_lookup("狂飙")
    all_ok &= check(len(tl) > 0, f"标题精确匹配《狂飙》命中 {len(tl)} 条")

    print("\n=== 4. Agentic RAG 问答 ===")
    graph = build_agentic_rag()
    queries = [
        "《狂飙》里高启强是什么人？张译演的谁？",
        "推荐几部评分高的科幻电影",
        "在哪里可以看《进击的巨人》？",
    ]
    for q_ in queries:
        print(f"\n  问: {q_}")
        out = graph.invoke(
            {
                "query": q_,
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
        answer = out["answer"].strip().replace("\n", "\n     ")
        print(f"  答: {answer[:600]}")
        print(f"  轨迹: {out['trace']}")
        all_ok &= check(bool(answer) and len(answer) > 20, f"问答有实质内容（{len(answer)}字）")

    print("\n=== 5. ReAct 多步智能体 ===")
    agent = build_react_agent()
    res = agent.invoke({"messages": [("user", "《凡人修仙传》是什么类型的动漫？在哪能看？")]})
    last = res["messages"][-1].content
    print(f"  答: {last.strip()[:400]}")
    all_ok &= check(bool(last) and len(last) > 20, "ReAct 智能体回答成功")
    tool_calls = sum(1 for m in res["messages"] if getattr(m, "tool_calls", None))
    all_ok &= check(tool_calls > 0, f"智能体实际调用了工具（{tool_calls} 次）")

    print("\n=== 6. 联网兜底（可选，网络波动时允许跳过） ===")
    try:
        from app.tools import web_search_duckduckgo
        r = web_search_duckduckgo("流浪地球2 在线观看", max_results=3)
        print(f"  联网搜索返回 {len(r)} 条: {[x['title'][:30] for x in r[:3]]}")
        all_ok &= check(len(r) > 0, "DuckDuckGo 搜索可用")
    except Exception as e:  # noqa: BLE001
        print(f"  联网搜索不可用（非致命）: {e}")

    print("\n=========================================")
    print("结果:", "全部通过 ✅" if all_ok else "存在失败项 ❌")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
