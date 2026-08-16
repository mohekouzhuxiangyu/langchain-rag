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
  .msg { padding: 12px 16px; border-radius: 12px; line-height: 1.7; font-size: 15px; word-break: break-word; }
  .user { background: #2563eb; align-self: flex-end; max-width: 85%; }
  .bot { background: #1e293b; border: 1px solid #334155; align-self: flex-start; max-width: 100%; }
  .msg p { margin: 6px 0; }
  .msg h2, .msg h3, .msg h4, .msg h5, .msg h6 { margin: 10px 0 6px; color: #f1f5f9; font-weight: 600; }
  .msg h2 { font-size: 17px; } .msg h3 { font-size: 16px; } .msg h4 { font-size: 15px; }
  .msg ul, .msg ol { margin: 6px 0 6px 22px; }
  .msg li { margin: 3px 0; }
  .msg a { color: #60a5fa; text-decoration: underline; word-break: break-all; }
  .msg a:hover { color: #93c5fd; }
  .msg code { background: #0f172a; border: 1px solid #334155; border-radius: 4px; padding: 1px 5px; font-size: 13px; }
  .msg pre { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px 12px; overflow-x: auto; margin: 8px 0; }
  .msg pre code { border: none; padding: 0; }
  .msg table { border-collapse: collapse; margin: 8px 0; width: 100%; font-size: 13px; }
  .msg th, .msg td { border: 1px solid #334155; padding: 5px 10px; text-align: left; }
  .msg th { background: #334155; font-weight: 600; }
  .meta { font-size: 12px; color: #64748b; margin-top: 8px; }
  .sources { margin-top: 10px; font-size: 13px; color: #94a3b8; border-top: 1px dashed #334155; padding-top: 8px; }
  .sources div { margin: 2px 0; }
  .sources a { color: #7dd3fc; }
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

// ---------- 轻量 Markdown 渲染（先转义，防 XSS） ----------
function inline(s) {
  const links = [];
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (m, t, u) => { links.push([t, u]); return '\\u0001L' + (links.length - 1) + '\\u0001'; });
  s = s.replace(/(https?:\/\/[^\s<>"'\u3000)，。；、）\]]+)/g, (m) => { links.push([m, m]); return '\\u0001L' + (links.length - 1) + '\\u0001'; });
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[\s(（])\*([^*\\n]+)\*(?=[\s).,，。;；:：、]|$)/g, '$1<em>$2</em>');
  s = s.replace(/\\u0001L(\\d+)\\u0001/g, (m, i) => { const [t, u] = links[+i]; return '<a href="' + u + '" target="_blank" rel="noopener">' + t + '</a>'; });
  return s;
}

function renderTable(rows) {
  let html = '<table>';
  rows.forEach((r, i) => {
    if (/^\|[\s:|-]+\|$/.test(r)) return;
    const cells = r.replace(/^\||\|$/g, '').split('|').map(c => c.trim());
    const tag = i === 0 ? 'th' : 'td';
    html += '<tr>' + cells.map(c => '<' + tag + '>' + inline(c) + '</' + tag + '>').join('') + '</tr>';
  });
  return html + '</table>';
}

function md(src) {
  const lines = src.split('\\n');
  let html = '';
  let list = null, para = [], table = null, inCode = false, codeBuf = [];
  const closeList = () => { if (list) { html += '</' + list + '>'; list = null; } };
  const closePara = () => { if (para.length) { html += '<p>' + para.map(inline).join('<br>') + '</p>'; para = []; } };
  const closeTable = () => { if (table) { html += renderTable(table); table = null; } };

  for (const raw of lines) {
    const t = raw.trim();
    if (t.startsWith('```')) {
      if (inCode) { html += '<pre><code>' + codeBuf.join('\\n') + '</code></pre>'; codeBuf = []; inCode = false; }
      else { closeList(); closePara(); closeTable(); inCode = true; }
      continue;
    }
    if (inCode) { codeBuf.push(raw); continue; }
    if (!t) { closeList(); closePara(); closeTable(); continue; }
    if (/^\|.*\|$/.test(t)) { closeList(); closePara(); (table = table || []).push(t); continue; }
    closeTable();
    let m;
    if ((m = t.match(/^#{1,6}\s+(.*)$/))) {
      closeList(); closePara();
      const lv = Math.min(m[0].match(/^#+/)[0].length + 1, 6);
      html += '<h' + lv + '>' + inline(m[1]) + '</h' + lv + '>';
      continue;
    }
    if (/^[-*·]\s+/.test(t)) {
      closePara();
      if (list !== 'ul') { closeList(); html += '<ul>'; list = 'ul'; }
      html += '<li>' + inline(t.replace(/^[-*·]\s+/, '')) + '</li>';
      continue;
    }
    if (/^\d+[.、]\s+/.test(t)) {
      closePara();
      if (list !== 'ol') { closeList(); html += '<ol>'; list = 'ol'; }
      html += '<li>' + inline(t.replace(/^\d+[.、]\s+/, '')) + '</li>';
      continue;
    }
    closeList();
    para.push(t);
  }
  if (inCode) html += '<pre><code>' + codeBuf.join('\\n') + '</code></pre>';
  closeList(); closePara(); closeTable();
  return html;
}

async function ask(text) {
  addMsg('user', esc(text));
  const botEl = addMsg('bot', '<p>思考中…</p>');
  btn.disabled = true;
  try {
    const r = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: text})
    });
    const data = await r.json();
    let html = md(esc(data.answer || '（无回答）'));
    if (data.sources && data.sources.length) {
      html += '<div class="sources"><b>来源：</b>';
      const seen = new Set();
      for (const s of data.sources) {
        const key = (s.type||'kb') + (s.media_id||'') + (s.title||'') + (s.url||'');
        if (seen.has(key)) continue; seen.add(key);
        if (s.type === 'web') {
          const u = s.url || '';
          const label = esc(s.title || u || '链接');
          html += '<div>🌐 ' + (u ? '<a href="' + esc(u) + '" target="_blank" rel="noopener">' + label + '</a>' : label) + '</div>';
        } else {
          html += '<div>📚 《' + esc(s.title||'') + '》(' + esc(s.type||'') + ') 段落:' + esc(s.section||'') + ' 相关度:' + (s.score||'') + '</div>';
        }
      }
      html += '</div>';
    }
    if (data.trace && data.trace.length) {
      html += '<div class="meta">' + data.trace.map(esc).join(' → ') + '</div>';
    }
    botEl.innerHTML = html;
  } catch (e) {
    botEl.innerHTML = '<p>请求失败：' + esc(String(e)) + '</p>';
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
addMsg('bot', '你好！我是影剧漫资源助手。可以问我：\\n· 《狂飙》高启强是什么人？\\n· 推荐几部高分科幻电影\\n· 好看的国漫有哪些\\n· 在哪里可以看《进击的巨人》\\n（资源类问题会直接给出可点击的观看/下载链接）');
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.environ.get("MEDIA_RAG_PORT", "8001"))
    uvicorn.run(app, host="127.0.0.1", port=port)
