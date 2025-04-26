from fastapi import FastAPI
from contextlib import asynccontextmanager
from core.model_loader import init_model
from fastapi.middleware.cors import CORSMiddleware
from api import predict, retrieval, health
from core.logger import get_logger

logger = get_logger(__name__)

import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing model...")
    init_model(os.getenv("MODEL_PATH", "model/finbert_v1-5e6_custom_eval"))
    logger.info("Model initialized successfully")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Economy Predictor API",
    description="경제 기사 분석 및 예측 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 적절한 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 연결
app.include_router(predict.router, prefix="/api", tags=["predict"]) 
app.include_router(retrieval.router, prefix="/api", tags=["retrieval"])
app.include_router(health.router, prefix="/api", tags=["health"])