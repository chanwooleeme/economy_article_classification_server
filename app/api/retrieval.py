from fastapi import APIRouter, Query
from services.qdrant_service import query_recent_articles, retrieve_articles_by_keywords, update_article_importance
from models.article import ArticleResponse, UpdateArticleImportanceRequest
from models.base_response import BaseResponseModel

router = APIRouter()

@router.get("/recent-articles", response_model=BaseResponseModel[ArticleResponse])
def recent_articles(top_k: int = Query(default=10, gt=0, le=100)):
    articles = query_recent_articles("econ_important", top_k)
    results = [ArticleResponse(id=a.id, content=a.content) for a in articles]
    return BaseResponseModel(results=results, count=len(results))

@router.get("/search-articles", response_model=BaseResponseModel[ArticleResponse])
def search_articles_by_keywords(time_range_sec: int = Query(default=0, gt=0), top_k: int = Query(default=10, gt=0, le=100)):
    results = [ArticleResponse(id=a['id'], content=a['content']) for a in retrieve_articles_by_keywords(time_range_sec, top_k)]
    return BaseResponseModel(results=results, count=len(results))

@router.post("/update-article-importance")
def update_importance(request: UpdateArticleImportanceRequest):
    result = update_article_importance(request.point_id, request.importance)
    if result:
        return BaseResponseModel(success=True)
    else:
        return BaseResponseModel(success=False)