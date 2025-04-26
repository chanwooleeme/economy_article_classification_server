from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone

class ArticleInput(BaseModel):
    title: str
    content: str
    category: str = ""
    author: str = ""
    custom_id: Optional[str] = None
    publication_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class InferenceRequest(BaseModel):
    articles: List[ArticleInput]

class ArticleResponse(BaseModel):
    id: str
    content: str


class Article(BaseModel):
    id: str
    content: str
    title: str
    author: str
    publication_date: datetime
    category: str

    
class UpdateArticleImportanceRequest(BaseModel):
    point_id: str
    importance: int