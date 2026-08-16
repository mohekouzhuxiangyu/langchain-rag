"""媒体数据模型。"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

MEDIA_TYPES = ("movie", "tv", "anime")
MEDIA_TYPE_LABEL = {"movie": "电影", "tv": "电视剧", "anime": "动漫"}


class MediaItem(BaseModel):
    """一条电影 / 电视剧 / 动漫的元数据（用于入库）。"""

    id: str = Field(description="唯一 id，如 movie_wandering_earth_2")
    title: str = Field(description="主标题（中文）")
    aliases: list[str] = Field(default_factory=list, description="别名 / 英文名 / 其他译名")
    type: str = Field(description="movie | tv | anime")
    year: int = Field(description="上映/首播年份")
    genres: list[str] = Field(default_factory=list, description="类型标签")
    rating: Optional[float] = Field(default=None, description="豆瓣/综合评分，0-10")
    director: str = Field(default="", description="导演 / 作者 / 原作")
    cast: list[str] = Field(default_factory=list, description="主演 / 声优")
    synopsis: str = Field(default="", description="剧情简介")
    episodes: Optional[int] = Field(default=None, description="集数（剧集/动漫）")
    status: str = Field(default="完结", description="连载状态")
    platform: str = Field(default="", description="国内正版播放平台")
    resource: str = Field(default="", description="常见资源/观看途径说明")
    tags: list[str] = Field(default_factory=list, description="额外标签，如 科幻/悬疑/国漫")
    awards: str = Field(default="", description="获奖情况")

    @property
    def type_label(self) -> str:
        return MEDIA_TYPE_LABEL.get(self.type, self.type)


class SearchHit(BaseModel):
    """一条检索结果。"""

    media_id: str
    title: str
    type: str
    section: str
    content: str
    score: float
    meta: dict[str, Any] = Field(default_factory=dict)

    def to_source(self) -> dict[str, Any]:
        return {
            "media_id": self.media_id,
            "title": self.title,
            "type": self.type,
            "section": self.section,
            "score": round(min(self.score, 1.0), 4),
            "meta": self.meta,
        }
