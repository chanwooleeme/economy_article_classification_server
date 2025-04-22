# ────────────────
# 1단계: 빌드 스테이지
# ────────────────
FROM python:3.9-slim as builder

WORKDIR /app

# 빌드용 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 프로젝트 복사
COPY app/ app/
COPY model/ model/
COPY setup.py .

# ⚡ torch는 먼저 설치해서 캐시 활용
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 나머지 패키지 설치
RUN pip install .

# ────────────────
# 2단계: 런타임 스테이지
# ────────────────
FROM python:3.9-slim

WORKDIR /app

# 빌드 단계에서 필요한 파일만 복사
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app/app /app/app
COPY --from=builder /app/model /app/model

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/model/finbert_v1-5e6_custom_eval

# 포트 열기
EXPOSE 8000

# FastAPI 서버 실행
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
