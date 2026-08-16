"""嵌入模型：本地 sentence-transformers 中文向量模型（经 HF 镜像下载）。"""
from __future__ import annotations

import os
from functools import lru_cache

from app.config import settings

os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


@lru_cache
def _load_model():
    from sentence_transformers import SentenceTransformer

    print(f"[embeddings] 加载本地嵌入模型 {settings.embedding_model} ...")
    model = SentenceTransformer(settings.embedding_model, trust_remote_code=True)
    model.eval()
    print("[embeddings] 模型就绪")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量文本 -> 向量。"""
    model = _load_model()
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    """单条查询 -> 向量（bge 建议查询侧加指令前缀，小模型可不加，保持与入库一致）。"""
    return embed_texts([text])[0]


def vector_literal(vec: list[float]) -> str:
    """pgvector 向量字面量（[a,b,c] 格式，psycopg2 直接可参数化）。"""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"
