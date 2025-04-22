from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pydantic import BaseModel, Field
from typing import List
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from datetime import datetime
import os

# ✅ 모델 & 토크나이저 로드 (FastAPI 실행 시 1회만 수행)
# 환경 변수에서 모델 경로 읽기 (기본값 설정)
model_path = os.getenv("MODEL_PATH", "model/finbert_v1-5e6_custom_eval")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

app = FastAPI(
    title="Economy Predictor API",
    description="경제 기사 분석 및 예측 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 적절한 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ArticleInput(BaseModel):
    title: str
    content: str
    category: str = ""
    author: str = ""
    custom_id: str = None
    publication_date: datetime = Field(default_factory=datetime.now)

class InferenceRequest(BaseModel):
    articles: List[ArticleInput]

def get_uuid_from_string(id_str: str) -> uuid.UUID:
    """문자열을 UUID로 변환하는 헬퍼 함수"""
    try:
        # 이미 UUID인 경우
        if isinstance(id_str, uuid.UUID):
            return id_str
        # UUID 문자열인 경우
        return uuid.UUID(id_str)
    except (ValueError, AttributeError):
        # 유효한 UUID가 아닌 경우 새로 생성
        return uuid.uuid4()

@app.post("/predict_and_store")
def predict_and_store(request: InferenceRequest):
    articles = []
    texts = []

    for article in request.articles:
        text = article.content.strip()
        tokens = tokenizer.tokenize(text)
        trimmed = tokenizer.convert_tokens_to_string(tokens[:512])
        texts.append(trimmed)

        # custom_id 처리
        article_id = get_uuid_from_string(article.custom_id) if article.custom_id else uuid.uuid4()
        
        articles.append({
            "title": article.title,
            "content": article.content,
            "category": article.category,
            "author": article.author,
            "custom_id": str(article_id),  # 문자열로 저장
            "publication_date": article.publication_date
        })

    embeddings = [] # 임베딩을 저장할 리스트
    # inference
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        # output_hidden_states=True 추가
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1).tolist()

        # Hidden states 추출 (마지막 레이어)
        last_hidden_states = outputs.hidden_states[-1] # shape: (batch_size, sequence_length, hidden_size)

        # Mean Pooling 수행
        # attention_mask를 사용하여 패딩 토큰은 제외하고 평균 계산
        input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        batch_embeddings = sum_embeddings / sum_mask
        embeddings = batch_embeddings.tolist() # 결과를 리스트로 변환

    # Qdrant 저장 시 위에서 계산한 embeddings 사용
    for idx, pred in enumerate(predictions):
        payload = articles[idx]
        payload["probability"] = probs[idx][pred].item()
        # get_embedding 대신 계산된 임베딩 사용
        embedding_vector = embeddings[idx]
        pt_id = payload["custom_id"]

        
        if pred == 1:
            qdrant_client.upsert(
                collection_name="econ_important",
                points=[PointStruct(id=pt_id, vector=embedding_vector, payload=payload)]
            )
        else:
            qdrant_client.upsert(
                collection_name="econ_not_important",
                points=[PointStruct(id=pt_id, vector=embedding_vector, payload=payload)]
            )

    return {
        "result": [
            {"custom_id": a["custom_id"], "label": p, "probability": probs[i][p].item()}
            for i, (a, p) in enumerate(zip(articles, predictions))
        ]
    } 