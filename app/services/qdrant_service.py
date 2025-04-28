import sys
import os
import time
from qdrant_client.models import PointStruct, Filter, Range, FieldCondition, PointIdsList
from core.model_loader import get_qdrant_client
from qdrant_client import QdrantClient
from models.article import Article
from typing import List, Optional
from core.logger import get_logger
from services.embedding_service import get_embeddings

logger = get_logger(__name__)

COLLECTION_NAME = "econ_important"
NON_IMPORTANT_COLLECTION_NAME = "econ_not_important"
ECON_KEYWORDS = [
    "금리", "환율", "부동산", "증시", "국제 유가", "정부 정책",
    "물가", "무역", "산업 정책", "전쟁", "중앙은행", "달러", "금",
    "고용", "실업률", "취업자 수",        # 노동 시장
    "GDP", "경제 성장", "성장률",          # 거시지표
    "소비", "소비자 심리", "소매 판매",     # 내수/소비
    "공급망", "반도체", "원자재"            # 산업/공급
]

def store_prediction_result(pred, payload: dict, vector: list):
    print(pred)
    if pred == 0:
        collection_name = NON_IMPORTANT_COLLECTION_NAME
    else:
        collection_name = COLLECTION_NAME

    get_qdrant_client().upsert(
        collection_name=collection_name,
        points=[PointStruct(id=payload["custom_id"], vector=vector, payload=payload)]
    )
    logger.info(f"Stored prediction result for {payload['custom_id']} in {COLLECTION_NAME}")


def query_recent_articles(collection_name: str, top_k: int = 10) -> List[Article]:
    qdrant_client = get_qdrant_client()
        
    now_ts = time.time()
    last_24_hours = now_ts - 24 * 60 * 60

    scroll_result = qdrant_client.scroll(
        collection_name=collection_name,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="publication_date",
                    range=Range(gte=last_24_hours)
                )
            ]
        )
    )

    # scroll_result: (List[ScoredPoint], Optional[ScrollId])
    points, _ = scroll_result
    logger.info(f"Found {len(points)} articles in {collection_name}")
    return [Article(
        id=p.id,
        content=p.payload['content'],
        title=p.payload['title'],
        author=p.payload['author'],
        publication_date=p.payload['publication_date'],
        category=p.payload['category']) for p in points]

    
def retrieve_articles_by_keywords(time_range_sec: int = 0, top_k: int = 10):
    embeddings = get_embeddings(ECON_KEYWORDS)
    qdrant_client = get_qdrant_client()
    if time_range_sec > 0:
        now_ts = time.time()
        last_time_range = now_ts - time_range_sec
        time_filter = Filter(
            must=[
                FieldCondition(
                    key="publication_date",
                    range=Range(gte=last_time_range)
                )
            ]
        )
    else:
        time_filter = None
    search_results = []
    for vector in embeddings:
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k,
            with_payload=True,
            query_filter=time_filter
        )
        search_results.extend(results)
    article_dict = dict()
    for result in search_results:
        if result[1] is not None:
            points = result[1]
            for p in points:
                if p.id not in article_dict and p.payload.get('content') is not None:
                    article_dict[p.id] = p.payload['content']
    
    final_list = [{"id": k, "content": v} for k, v in article_dict.items()]
    logger.info(f"Found {len(final_list)} articles")
    return final_list


def update_article_importance(point_id: str, importance: int):
    qdrant_client = get_qdrant_client()
    meta_data = {
        'importance': importance
    }
    try:
        qdrant_client.set_payload(
            collection_name=COLLECTION_NAME,
            points=[point_id],
            payload=meta_data
        )
        return True
    except Exception as e:
        logger.error(f"Error updating article importance: {e}")
        return False