#!/bin/bash

# 경제 예측 API 배포 스크립트
# 사용법: ./deploy.sh [--build] [--no-cache]
# 옵션:
#   --build      Docker 이미지를 새로 빌드합니다
#   --no-cache   Docker 이미지 빌드 시 캐시를 사용하지 않습니다

set -e

# 현재 위치 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 기본 설정
CONTAINER_NAME="economy-predictor"
IMAGE_NAME="economy-predictor"
BUILD=false
NO_CACHE=""

# 변수 파싱
for arg in "$@"; do
  case $arg in
    --build)
      BUILD=true
      shift
      ;;
    --no-cache)
      NO_CACHE="--no-cache"
      shift
      ;;
  esac
done

# 로고 출력
echo "==========================================="
echo "    경제 예측 API 서버 배포 스크립트    "
echo "==========================================="

# 기존 컨테이너 확인 및 제거
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "🔄 기존 컨테이너 중지 및 제거 중..."
    docker stop $CONTAINER_NAME > /dev/null && docker rm $CONTAINER_NAME > /dev/null || true
elif [ "$(docker ps -aq -f status=exited -f name=$CONTAINER_NAME)" ]; then
    echo "🧹 종료된 컨테이너 제거 중..."
    docker rm $CONTAINER_NAME > /dev/null || true
fi

# 빌드 옵션이 활성화된 경우 Docker 이미지 빌드
if [ "$BUILD" = true ]; then
    echo "🔨 Docker 이미지 빌드 중..."
    docker build $NO_CACHE -t $IMAGE_NAME .
fi

# 모델 경로 확인
MODEL_PATH="$SCRIPT_DIR/model/finbert_v1-5e6_custom_eval"
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 오류: 모델 디렉토리를 찾을 수 없습니다: $MODEL_PATH"
    exit 1
fi

# 환경 변수 파일 확인
ENV_FILE="$SCRIPT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ 오류: .env 파일을 찾을 수 없습니다: $ENV_FILE"
    exit 1
fi

# Docker 컨테이너 실행
echo "🚀 API 서버 시작 중..."
docker run -d \
    --name $CONTAINER_NAME \
    -p 8000:8000 \
    --env-file $ENV_FILE \
    --volume "$MODEL_PATH:/app/models/finbert" \
    --restart unless-stopped \
    $IMAGE_NAME

# 상태 확인
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "✅ API 서버가 성공적으로 배포되었습니다!"
    echo "📝 로그 확인: docker logs -f $CONTAINER_NAME"
    echo "🔗 API 접속: http://localhost:8000"
    echo "💻 컨테이너 중지: docker stop $CONTAINER_NAME"
else
    echo "❌ API 서버 배포 실패. 로그를 확인해주세요."
    docker logs $CONTAINER_NAME || echo "로그를 불러올 수 없습니다."
fi 