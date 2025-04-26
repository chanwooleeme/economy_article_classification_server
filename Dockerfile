# ────────────────
# 1단계: 빌드 스테이지
# ────────────────
FROM python:3.9-slim as builder

WORKDIR /app

# 빌드용 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 프로젝트 복사 (모델 제외)
COPY app/ /app/app/
COPY setup.py .

# ⚡ torch는 먼저 설치해서 캐시 활용
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 나머지 패키지 설치
RUN pip install .

# 추가 필요 패키지 설치
RUN pip install colorlog

# ────────────────
# 2단계: 런타임 스테이지
# ────────────────
FROM python:3.9-slim

WORKDIR /app

# 빌드 단계에서 필요한 파일만 복사
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
# app 폴더 전체 복사
COPY --from=builder /app/app /app/app

# 모델 마운트를 위한 디렉토리 생성
RUN mkdir -p /app/models

# 환경 변수 설정
ENV PYTHONPATH=/app:/app/app
# 볼륨으로 마운트될 모델 경로 설정
ENV MODEL_PATH=/app/models/finbert

# 포트 열기
EXPOSE 8000

# FastAPI 서버 실행
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
