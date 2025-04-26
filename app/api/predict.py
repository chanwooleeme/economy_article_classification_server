from fastapi import APIRouter
from models.article import InferenceRequest
from services.prediction_service import predict_articles
from services.qdrant_service import store_prediction_result
import uuid

router = APIRouter()

def get_uuid_from_string(id_str: str) -> uuid.UUID:
    try:
        return uuid.UUID(id_str)
    except:
        return uuid.uuid4()

@router.post("/predict")
def predict(request: InferenceRequest):
    articles = []
    texts = []

    for article in request.articles:
        text = article.content.strip()
        texts.append(text)

        # custom_id 처리
        article_id = get_uuid_from_string(article.custom_id) if article.custom_id else uuid.uuid4()
        
        articles.append({
            "title": article.title,
            "content": article.content,
            "category": article.category,
            "author": article.author,
            "custom_id": str(article_id),
            "publication_date": article.publication_date.timestamp()
        })

    predictions, probs, embeddings = predict_articles(texts)

    results = []
    for idx, pred in enumerate(predictions):
        payload = articles[idx]
        payload["probability"] = probs[idx][pred].item()

        store_prediction_result(pred, payload, embeddings[idx])
        results.append({
            "custom_id": payload["custom_id"],
            "label": pred,
            "probability": payload["probability"]
        })

    return {"results": results}
    