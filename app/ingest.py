"""入库流水线：媒体元数据 -> 分块 -> 向量化 -> upsert 进 PostgreSQL(pgvector)。"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Iterable

import psycopg2
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.db import init_db
from app.embeddings import embed_texts, vector_literal
from app.models import MediaItem

COLLECTION_NAME = "media_collection"


def _conn():
    return psycopg2.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        user=settings.pg_user,
        password=settings.pg_password,
        dbname=settings.pg_database,
    )


def _collection_id(conn) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s", (COLLECTION_NAME,))
        row = cur.fetchone()
        if not row:
            raise RuntimeError("collection 不存在，请先执行 app.db.init_db()")
        return str(row[0])


def build_chunks(item: MediaItem) -> list[Document]:
    """把一个媒体条目切成多个带元数据的 Document。"""
    docs: list[Document] = []
    title_line = f"{item.title}（{item.year}）"
    type_label = item.type_label
    genres = "/".join(item.genres) or "未知"
    rating = f"{item.rating:.1f}" if item.rating else "暂无"
    meta_base = {
        "media_id": item.id,
        "title": item.title,
        "type": item.type,
        "year": item.year,
        "rating": item.rating,
        "genres": item.genres,
    }

    def add(section: str, text: str, extra: dict | None = None):
        meta = {**meta_base, "section": section}
        if extra:
            meta.update(extra)
        docs.append(Document(page_content=text, metadata=meta))

    # 1) 总览块
    overview = (
        f"{title_line} 类型：{type_label}；类型标签：{genres}；评分：{rating}。\n"
        f"剧情简介：{item.synopsis}"
    )
    if item.aliases:
        overview += f"\n别名/其他译名：{'、'.join(item.aliases)}"
    if item.tags:
        overview += f"\n特色标签：{'、'.join(item.tags)}"
    if len(overview) <= 600:
        add("overview", overview)
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=80, separators=["\n", "。", "；", "，", " "]
        )
        head, *rest = splitter.split_text(overview)
        add("overview", head)
        for i, piece in enumerate(rest, 1):
            add(f"overview_{i}", piece)

    # 2) 主创/演员块
    if item.director or item.cast:
        add(
            "cast",
            f"{title_line} 导演/作者：{item.director or '未知'}；"
            f"主演/声优：{'、'.join(item.cast) if item.cast else '未知'}。",
        )

    # 3) 资源/观看途径块（本系统核心用途：查资源）
    if item.platform or item.resource:
        add(
            "resource",
            f"{title_line} 正版播放平台：{item.platform or '未收录'}。"
            f"资源/观看途径：{item.resource or '未收录'}。",
        )

    # 4) 集数/连载状态块
    if item.episodes or item.status:
        add(
            "status",
            f"{title_line} 集数：{item.episodes if item.episodes else '未知'} 集；"
            f"连载状态：{item.status}。",
        )

    # 5) 获奖块
    if item.awards:
        add("awards", f"{title_line} 获奖情况：{item.awards}")

    return docs


def upsert_media(items: Iterable[MediaItem]) -> dict:
    """入库：先写 media_items 主表，再重写向量块。返回统计。"""
    init_db()
    items = list(items)
    conn = _conn()
    conn.autocommit = True
    cid = _collection_id(conn)

    # --- 1) 主表 upsert ---
    with conn.cursor() as cur:
        for it in items:
            cur.execute(
                """
                INSERT INTO media_items
                    (id, title, aliases, type, year, genres, rating, director, cast_list,
                     synopsis, episodes, status, platform, resource, tags, awards, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
                ON CONFLICT (id) DO UPDATE SET
                    title=EXCLUDED.title, aliases=EXCLUDED.aliases, type=EXCLUDED.type,
                    year=EXCLUDED.year, genres=EXCLUDED.genres, rating=EXCLUDED.rating,
                    director=EXCLUDED.director, cast_list=EXCLUDED.cast_list, synopsis=EXCLUDED.synopsis,
                    episodes=EXCLUDED.episodes, status=EXCLUDED.status,
                    platform=EXCLUDED.platform, resource=EXCLUDED.resource,
                    tags=EXCLUDED.tags, awards=EXCLUDED.awards, updated_at=now()
                """,
                (
                    it.id, it.title, json.dumps(it.aliases, ensure_ascii=False), it.type,
                    it.year, json.dumps(it.genres, ensure_ascii=False), it.rating,
                    it.director, json.dumps(it.cast, ensure_ascii=False), it.synopsis,
                    it.episodes, it.status, it.platform, it.resource,
                    json.dumps(it.tags, ensure_ascii=False), it.awards,
                ),
            )

    # --- 2) 向量块重建（幂等：先删该 media 旧块） ---
    all_docs: list[Document] = []
    for it in items:
        all_docs.extend(build_chunks(it))

    with conn.cursor() as cur:
        for it in items:
            cur.execute(
                "DELETE FROM langchain_pg_embedding WHERE collection_id=%s AND cmetadata->>'media_id'=%s",
                (cid, it.id),
            )

    if all_docs:
        texts = [d.page_content for d in all_docs]
        print(f"[ingest] 向量化 {len(texts)} 个文本块 ...")
        vectors = embed_texts(texts)
        with conn.cursor() as cur:
            for doc, vec in zip(all_docs, vectors):
                cur.execute(
                    """
                    INSERT INTO langchain_pg_embedding
                        (collection_id, embedding, document, cmetadata, uuid)
                    VALUES (%s, %s::vector, %s, %s, %s)
                    """,
                    (cid, vector_literal(vec), doc.page_content, json.dumps(doc.metadata, ensure_ascii=False), str(uuid.uuid4())),
                )
    conn.close()
    return {"media": len(items), "chunks": len(all_docs)}


def load_items_from_json(path: str | Path) -> list[MediaItem]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [MediaItem(**d) for d in raw]


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "data/samples.json"
    items = load_items_from_json(src)
    stat = upsert_media(items)
    print(f"[ingest] 完成：入库媒体 {stat['media']} 条，向量块 {stat['chunks']} 个")
