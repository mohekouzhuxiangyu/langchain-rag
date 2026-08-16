"""FastAPI Web 服务：影剧漫 Agentic RAG 知识库 HTTP 接口 + 简易聊天页。

启动：
    uvicorn app.server:app --host 127.0.0.1 --port 8001
或：
    python -m app.server
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from app.graph import build_agentic_rag, run_agentic_rag
from app.retriever import hybrid_search

app = FastAPI(title="影剧漫 Agentic RAG 知识库", version="1.0.0", description="电影/电视剧/动漫资源查询")


class AskRequest(BaseModel):
    query: str
    agent: bool = False


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    trace: list[str]
    retrieval: list[dict] = []


@lru_cache
def _graph():
    return build_agentic_rag()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


@app.get("/api/health")
def health() -> dict:
    try:
        from app.ingest import _conn

        conn = _conn()
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM media_items")
            media = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM langchain_pg_embedding")
            chunks = cur.fetchone()[0]
        conn.close()
        return {"status": "ok", "media_items": media, "vector_chunks": chunks, "llm": "deepseek-chat"}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not req.query.strip():
        return AskResponse(answer="请输入问题。", sources=[], trace=[])
    if req.agent:
        from app.graph import build_react_agent

        agent = build_react_agent()
        result = agent.invoke({"messages": [("user", req.query)]})
        return AskResponse(
            answer=result["messages"][-1].content,
            sources=[],
            trace=["ReAct 智能体模式"],
        )
    out = _graph().invoke(
        {
            "query": req.query,
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
    return AskResponse(answer=out["answer"], sources=out["sources"], trace=out["trace"])


@app.get("/api/retrieve")
def retrieve(query: str, media_type: str | None = None) -> list[dict]:
    hits = hybrid_search(query, media_type=media_type, top_k=5)
    return [h.to_source() | {"content": h.content[:200]} for h in hits]


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>影剧漫 Agentic RAG 知识库</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         background: #0f172a; color: #e2e8f0; min-height: 100vh; display: flex; flex-direction: column; }
  header { padding: 18px 24px; background: #1e293b; border-bottom: 1px solid #334155;
           display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; font-weight: 600; }
  header .badge { font-size: 12px; color: #94a3b8; background: #334155; padding: 3px 10px; border-radius: 999px; }
  main { flex: 1; max-width: 860px; width: 100%; margin: 0 auto; padding: 24px 16px;
         display: flex; flex-direction: column; gap: 16px; overflow-y: auto; }
  .msg { padding: 12px 16px; border-radius: 12px; line-height: 1.7; font-size: 15px; white-space: pre-wrap; word-break: break-word; }
  .user { background: #2563eb; align-self: flex-end; max-width: 85%; }
  .bot { background: #1e293b; border: 1px solid #334155; align-self: flex-start; max-width: 100%; }
  .meta { font-size: 12px; color: #64748b; margin-top: 8px; }
  .sources { margin-top: 10px; font-size: 13px; color: #94a3b8; border-top: 1px dashed #334155; padding-top: 8px; }
  .sources div { margin: 2px 0; }
  form { display: flex; gap: 10px; padding: 16px; background: #1e293b; border-top: 1px solid #334155; }
  input[type=text] { flex: 1; padding: 12px 16px; border-radius: 10px; border: 1px solid #334155;
                     background: #0f172a; color: #e2e8f0; font-size: 15px; outline: none; }
  input[type=text]:focus { border-color: #2563eb; }
  button { padding: 12px 24px; border-radius: 10px; border: none; background: #2563eb; color: #fff;
           font-size: 15px; cursor: pointer; }
  button:disabled { opacity: .5; cursor: wait; }
  .hint { font-size: 12px; color: #64748b; text-align: center; padding-bottom: 8px; }
</style>
</head>
<body>
<header>
  <h1>🎬 影剧漫 Agentic RAG 知识库</h1>
  <span class="badge">LangChain + LangGraph + PostgreSQL/pgvector + DeepSeek</span>
</header>
<main id="chat"></main>
<form id="form">
  <input type="text" id="q" placeholder="问电影/电视剧/动漫：剧情、演员、评分、推荐、在哪看…" autocomplete="off">
  <button id="btn" type="submit">发送</button>
</form>
<div class="hint">知识库无相关内容时自动联网搜索兜底 · 支持「在哪看」「推荐」「是谁演的」等问题</div>
<script>
const chat = document.getElementById('chat');
const form = document.getElementById('form');
const q = document.getElementById('q');
const btn = document.getElementById('btn');

function addMsg(role, html) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = html;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}
function esc(s) { return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

async function ask(text) {
  addMsg('user', esc(text));
  const botEl = addMsg('bot', '思考中…');
  btn.disabled = true;
  try {
    const r = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: text})
    });
    const data = await r.json();
    let html = esc(data.answer || '（无回答）');
    if (data.sources && data.sources.length) {
      html += '<div class="sources"><b>来源：</b>';
      const seen = new Set();
      for (const s of data.sources) {
        const key = (s.type||'kb') + (s.media_id||'') + (s.title||'') + (s.url||'');
        if (seen.has(key)) continue; seen.add(key);
        if (s.type === 'web') html += '<div>🌐 ' + esc(s.title || '') + ' ' + esc(s.url || '') + '</div>';
        else html += '<div>📚 《' + esc(s.title||'') + '》(' + esc(s.type||'') + ') 段落:' + esc(s.section||'') + ' 相关度:' + (s.score||'') + '</div>';
      }
      html += '</div>';
    }
    if (data.trace && data.trace.length) {
      html += '<div class="meta">' + data.trace.map(esc).join(' → ') + '</div>';
    }
    botEl.innerHTML = html;
  } catch (e) {
    botEl.innerHTML = '请求失败：' + esc(String(e));
  }
  btn.disabled = false;
  q.focus();
}

form.addEventListener('submit', e => {
  e.preventDefault();
  const text = q.value.trim();
  if (!text) return;
  q.value = '';
  ask(text);
});
addMsg('bot', '你好！我是影剧漫资源助手。可以问我：\\n· 《狂飙》高启强是什么人？\\n· 推荐几部高分科幻电影\\n· 好看的国漫有哪些\\n· 在哪里可以看《进击的巨人》');
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.environ.get("MEDIA_RAG_PORT", "8001"))
    uvicorn.run(app, host="127.0.0.1", port=port)
