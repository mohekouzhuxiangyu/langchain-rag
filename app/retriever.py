"""混合检索：pgvector 向量相似度 + PostgreSQL 全文检索 + 标题精确匹配，RRF 融合。"""
from __future__ import annotations

import json
import re
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor, register_default_jsonb

from app.config import settings
from app.embeddings import embed_query, vector_literal
from app.models import MEDIA_TYPES, SearchHit

COLLECTION_NAME = "media_collection"
_PUNCT = re.compile(r"[^\w\u4e00-\u9fff]+")


def _conn():
    conn = psycopg2.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        user=settings.pg_user,
        password=settings.pg_password,
        dbname=settings.pg_database,
        cursor_factory=RealDictCursor,
    )
    register_default_jsonb(conn_or_curs=conn, globally=False, loads=json.loads)
    return conn


def _tokenize(text: str) -> list[str]:
    return [t for t in _PUNCT.sub(" ", text.lower()).split() if t]


def title_lookup(query: str) -> list[dict]:
    """在 media_items 主表中做标题/别名匹配，返回命中的媒体元数据。"""
    toks = _tokenize(query)
    if not toks:
        return []
    conn = _conn()
    rows = []
    with conn.cursor() as cur:
        # 中文标题直接 LIKE 匹配；英文按词匹配
        like = f"%{query.strip()}%"
        cur.execute(
            """
            SELECT * FROM media_items
            WHERE title LIKE %s OR aliases::text ILIKE %s
            ORDER BY CASE WHEN title = %s THEN 0 WHEN title LIKE %s THEN 1 ELSE 2 END
            LIMIT 5
            """,
            (like, like, query.strip(), f"{query.strip()}%"),
        )
        rows = list(cur.fetchall())
        if not rows and toks:
            cond = " OR ".join(["title ILIKE %s"] * len(toks))
            cur.execute(f"SELECT * FROM media_items WHERE {cond} LIMIT 5", [f"%{t}%" for t in toks])
            rows = list(cur.fetchall())
    conn.close()
    for r in rows:
        for k in ("aliases", "genres", "cast_list", "tags"):
            r[k] = r[k] if isinstance(r[k], list) else []
        r["cast"] = r.pop("cast_list", [])
    return rows


def _vector_search(conn, cid: str, query_vec: list[float], media_type: Optional[str], limit: int) -> dict[str, dict]:
    vec_lit = vector_literal(query_vec)
    sql = """
        SELECT uuid::text, document, cmetadata, 1 - (embedding <=> %s::vector) AS sim
        FROM langchain_pg_embedding
        WHERE collection_id = %s
    """
    params: list = [vec_lit, cid]
    if media_type in MEDIA_TYPES:
        sql += " AND cmetadata->>'type' = %s"
        params.append(media_type)
    sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
    params += [vec_lit, limit]
    out: dict[str, dict] = {}
    with conn.cursor() as cur:
        cur.execute(sql, params)
        for row in cur.fetchall():
            out[str(row["uuid"])] = {"doc": row["document"], "meta": row["cmetadata"], "sim": float(row["sim"])}
    return out


def _fts_search(conn, cid: str, terms: list[str], media_type: Optional[str], limit: int) -> dict[str, dict]:
    """关键词检索：document 直接 LIKE + to_tsvector('simple')（主要增强英文/数字词）。"""
    out: dict[str, dict] = {}
    with conn.cursor() as cur:
        sql = """
            SELECT uuid::text, document, cmetadata, ts_rank(to_tsvector('simple', document), plainto_tsquery('simple', %s)) AS rank
            FROM langchain_pg_embedding
            WHERE collection_id = %s AND to_tsvector('simple', document) @@ plainto_tsquery('simple', %s)
        """
        params: list = [" ".join(terms), cid, " ".join(terms)]
        if media_type in MEDIA_TYPES:
            sql += " AND cmetadata->>'type' = %s"
            params.append(media_type)
        sql += " ORDER BY rank DESC LIMIT %s"
        params.append(limit)
        cur.execute(sql, params)
        for row in cur.fetchall():
            out[str(row["uuid"])] = {"doc": row["document"], "meta": row["cmetadata"], "rank": float(row["rank"])}
        # 兜底：任一命中词即算（plainto_tsquery 对中文无效时用 LIKE）
        if not out:
            like_sql = f"""
                SELECT uuid::text, document, cmetadata, 1.0 AS rank
                FROM langchain_pg_embedding
                WHERE collection_id = %s AND ({" OR ".join(["document ILIKE %s"] * len(terms))})
            """
            like_params: list = [cid] + [f"%{t}%" for t in terms]
            if media_type in MEDIA_TYPES:
                like_sql += " AND cmetadata->>'type' = %s"
                like_params.append(media_type)
            like_sql += " LIMIT %s"
            like_params.append(limit)
            cur.execute(like_sql, like_params)
            for row in cur.fetchall():
                out[str(row["uuid"])] = {"doc": row["document"], "meta": row["cmetadata"], "rank": float(row["rank"])}
    return out


def hybrid_search(
    query: str,
    media_type: Optional[str] = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[SearchHit]:
    """向量 + 关键词 + 标题匹配 的混合检索。"""
    top_k = top_k or settings.top_k
    score_threshold = score_threshold if score_threshold is not None else settings.score_threshold
    conn = _conn()
    cid = None
    with conn.cursor() as cur:
        cur.execute("SELECT uuid::text FROM langchain_pg_collection WHERE name = %s", (COLLECTION_NAME,))
        row = cur.fetchone()
        cid = row["uuid"] if row else None
    if not cid:
        conn.close()
        return []

    # 标题匹配：命中则给对应媒体所有块加权
    title_hits = title_lookup(query)
    title_ids = {h["id"] for h in title_hits}
    boost: dict[str, float] = {}
    if title_ids:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT uuid::text, cmetadata FROM langchain_pg_embedding WHERE collection_id=%s AND cmetadata->>'media_id' = ANY(%s)",
                (cid, list(title_ids)),
            )
            for row in cur.fetchall():
                boost[str(row["uuid"])] = 0.25

    query_vec = embed_query(query)
    terms = _tokenize(query)

    vec_hits = _vector_search(conn, cid, query_vec, media_type, top_k * 4)
    fts_hits = _fts_search(conn, cid, terms, media_type, top_k * 4) if terms else {}
    conn.close()

    # 融合打分：0.7*向量相似度 + 0.3*FTS 排名分 + 标题命中加成
    vec_ranked = sorted(vec_hits, key=lambda u: -vec_hits[u]["sim"])
    fts_ranked = sorted(fts_hits, key=lambda u: -fts_hits[u]["rank"])
    fts_rank_of = {uid: i for i, uid in enumerate(fts_ranked)}

    merged: dict[str, float] = {}
    for rank, uid in enumerate(vec_ranked):
        sim = vec_hits[uid]["sim"]
        fts_part = 1.0 / (1 + fts_rank_of[uid]) if uid in fts_rank_of else 0.0
        merged[uid] = 0.7 * sim + 0.3 * fts_part + boost.get(uid, 0.0)
    for rank, uid in enumerate(fts_ranked):
        if uid not in merged:
            merged[uid] = 0.3 * (1.0 / (1 + rank)) + boost.get(uid, 0.0)

    ranked = sorted(merged.items(), key=lambda x: -x[1])

    results: list[SearchHit] = []
    for uid, score in ranked[: top_k * 2]:
        hit = vec_hits.get(uid) or fts_hits.get(uid)
        if not hit:
            continue
        meta = hit["meta"] or {}
        results.append(
            SearchHit(
                media_id=meta.get("media_id", ""),
                title=meta.get("title", ""),
                type=meta.get("type", ""),
                section=meta.get("section", ""),
                content=hit["doc"],
                score=score,
                meta=meta,
            )
        )
    results = [r for r in results if r.score >= score_threshold]
    return results[:top_k]
