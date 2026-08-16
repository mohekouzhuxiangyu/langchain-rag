"""PostgreSQL 初始化：建库、启用 pgvector、建表。"""
from __future__ import annotations

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from app.config import settings

# langchain PGVector 使用的表
LANCHAIN_TABLES = """
CREATE TABLE IF NOT EXISTS langchain_pg_collection (
    name VARCHAR (2048),
    cmetadata JSONB,
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    CONSTRAINT langchain_pg_collection_name_unique UNIQUE (name)
);
CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
    collection_id UUID,
    embedding VECTOR,
    document VARCHAR,
    cmetadata JSONB,
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid()
);
"""

# 媒体主表：结构化元数据，供标题精确匹配 / 全文检索增强
MEDIA_TABLE = """
CREATE TABLE IF NOT EXISTS media_items (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    aliases JSONB DEFAULT '[]',
    type TEXT NOT NULL CHECK (type IN ('movie','tv','anime')),
    year INT,
    genres JSONB DEFAULT '[]',
    rating REAL,
    director TEXT DEFAULT '',
    cast_list JSONB DEFAULT '[]',
    synopsis TEXT DEFAULT '',
    episodes INT,
    status TEXT DEFAULT '完结',
    platform TEXT DEFAULT '',
    resource TEXT DEFAULT '',
    tags JSONB DEFAULT '[]',
    awards TEXT DEFAULT '',
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_media_type ON media_items (type);
CREATE INDEX IF NOT EXISTS idx_media_title ON media_items (title);
"""


def _conn(database: str | None = None) -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        user=settings.pg_user,
        password=settings.pg_password,
        dbname=database or settings.pg_database,
    )


def ensure_database() -> None:
    """若目标库不存在则创建。"""
    conn = _conn("postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (settings.pg_database,))
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE "{settings.pg_database}"')
            print(f"[db] 已创建数据库 {settings.pg_database}")
    conn.close()


def ensure_pgvector() -> None:
    """启用 pgvector 扩展并建表。"""
    conn = _conn()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")  # gen_random_uuid
        cur.execute(LANCHAIN_TABLES)
        cur.execute(MEDIA_TABLE)
        cur.execute(
            "SELECT count(*) FROM langchain_pg_collection WHERE name = %s",
            ("media_collection",),
        )
        if cur.fetchone()[0] == 0:
            cur.execute(
                "INSERT INTO langchain_pg_collection (name, cmetadata) VALUES (%s, %s)",
                ("media_collection", '{"description": "movie/tv/anime resource RAG"}'),
            )
    conn.close()
    print("[db] pgvector 就绪：向量表 + media_items 主表 已创建/已存在")


def init_db() -> None:
    ensure_database()
    ensure_pgvector()


if __name__ == "__main__":
    init_db()
