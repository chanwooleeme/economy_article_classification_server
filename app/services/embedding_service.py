import torch
import os
from transformers import AutoTokenizer, AutoModel
from contextlib import contextmanager
from core.model_loader import get_model, get_tokenizer_pool

def get_embedding(text: str) -> list:
    """텍스트의 임베딩 벡터 생성"""
    with get_tokenizer_pool().use() as tokenizer:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        model = get_model()
        
        # 분류 모델에서 베이스 모델 추출
        base_model = model.base_model
        
        with torch.no_grad():
            # 분류 헤드 없이 기본 BERT 모델만 실행
            outputs = base_model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            return cls_emb.squeeze().tolist()

def get_embeddings(texts: list[str]) -> list[list]:
    """여러 텍스트의 임베딩 벡터 생성"""
    return [get_embedding(text) for text in texts]

if __name__ == "__main__":
    print(get_embedding("경제"))