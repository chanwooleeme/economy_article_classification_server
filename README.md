# Economy Predictor

경제 지표 예측을 위한 머신러닝 기반 프로젝트입니다.

## 프로젝트 구조

```
economy_predictor/
├── app/                    # 애플리케이션 코드
├── economy_predictor/      # 프로젝트 패키지
├── tests/                  # 테스트 코드
├── requirements.txt        # Python 의존성
├── pyproject.toml         # 프로젝트 설정
└── Dockerfile             # Docker 설정
```

## 개발 환경 설정

1. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
.\venv\Scripts\activate  # Windows
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
python -m economy_predictor
```

## 테스트 실행

```bash
pytest
```

## Docker 실행

```bash
docker build -t economy-predictor .
docker run -p 8000:8000 economy-predictor
```
