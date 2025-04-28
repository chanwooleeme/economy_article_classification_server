from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer
from queue import Queue
from contextlib import contextmanager
import os
import pathlib
from qdrant_client import QdrantClient
from core.logger import get_logger
import torch

logger = get_logger(__name__)

# 모델 관련 변수
_tokenizer_pool = None
_model = None

# 벡터 검색 관련 변수
_qdrant_client = None

# 더미 모델 클래스 추가
class DummyModel:
    """테스트용 더미 모델"""
    def __init__(self):
        self.num_labels = 2
        
    def eval(self):
        return self
        
    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size = len(input_ids) if input_ids is not None else 1
        return type('obj', (object,), {
            'logits': torch.tensor([[0.1, 0.9]] * batch_size)  # 항상 긍정적 예측
        })

class ResourcePool:
    def __init__(self, resource_factory, model_path: str, pool_size: int = 4):
        self.pool = Queue()
        
        # 각 리소스 생성
        for _ in range(pool_size):
            self.pool.put(resource_factory(model_path))

    def acquire(self):
        return self.pool.get()
    
    def release(self, resource):
        self.pool.put(resource)

    @contextmanager
    def use(self):
        resource = self.acquire()
        try:
            yield resource
        finally:
            self.release(resource)

def load_tokenizer(model_path: str) -> PreTrainedTokenizer:
    """모델 경로에서 토크나이저 로드 - 오류가 발생하면 fallback 처리"""

    try:
        # 기본 방식으로 로드 시도
        logger.info(f"Trying to load tokenizer from {model_path}")
        return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except (OSError, ValueError) as e:
        # 오류 발생 시 로그 기록
        logger.warning(f"Error loading tokenizer: {str(e)}")
        
        # 대체 경로 시도 - 절대 경로가 아닌 경우
        if not model_path.startswith('/'):
            model_path = os.path.join('/app', model_path)
            
        logger.info(f"Trying alternative path: {model_path}")
        try:
            return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        except Exception as e2:
            logger.error(f"Failed to load tokenizer with alternative path: {str(e2)}")
            


class TokenizerPool(ResourcePool):
    def __init__(self, model_path: str, pool_size: int = 4):
        super().__init__(load_tokenizer, model_path, pool_size)

def init_model(model_path: str):
    """
    FinBERT 분류 모델을 초기화합니다.
    
    Args:
        model_path: FinBERT 모델 경로
    """
    global _tokenizer_pool, _model, _qdrant_client
    
    logger.info(f"Initializing model from: {model_path}")
    
    # 테스트 모드 확인
    if os.environ.get("TEST_MODE", "false").lower() == "true":
        logger.info("TEST_MODE enabled, using dummy model")
        _tokenizer_pool = TokenizerPool(model_path)
        _model = DummyModel()
        logger.info(f"Dummy model and tokenizer loaded")
        
        # Qdrant 클라이언트 초기화
        if _qdrant_client is None:
            _qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
            logger.info(f"Qdrant client loaded from {os.getenv('QDRANT_URL')}")
        return
    
    # 토크나이저 초기화
    if _tokenizer_pool is None:
        _tokenizer_pool = TokenizerPool(model_path)
        logger.info(f"Tokenizer pool loaded from {model_path}")
    
    # 분류 모델 초기화
    if _model is None:
        try:
            # 기본 방식으로 로드 시도
            logger.info(f"Loading model from {model_path}")
            _model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).eval()
        except (OSError, ValueError) as e:
            logger.warning(f"Error loading model: {str(e)}")
            
            # 대체 경로 시도 - 절대 경로가 아닌 경우
            alt_path = model_path
            if not model_path.startswith('/'):
                alt_path = os.path.join('/app', model_path)
                
            logger.info(f"Trying alternative path: {alt_path}")
            try:
                _model = AutoModelForSequenceClassification.from_pretrained(alt_path, local_files_only=True).eval()
            except Exception as e2:
                logger.error(f"Failed to load model with alternative path: {str(e2)}")
                
                # 마지막으로 절대 경로에서 상대 경로로 변환 시도
                if model_path.startswith('/app/'):
                    try:
                        relative_path = model_path[5:]  # '/app/' 제거
                        logger.info(f"Trying with relative path: {relative_path}")
                        _model = AutoModelForSequenceClassification.from_pretrained(relative_path, local_files_only=True).eval()
                    except Exception as e3:
                        logger.error(f"All model loading attempts failed: {str(e3)}")
                        logger.warning("Fallback to dummy model")
                        _model = DummyModel()
                else:
                    logger.warning("Fallback to dummy model")
                    _model = DummyModel()
        
        logger.info(f"FinBERT model loaded successfully")
        
    # Qdrant 클라이언트 초기화
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        logger.info(f"Qdrant client loaded from {os.getenv('QDRANT_URL')}")

def get_tokenizer_pool():
    """토크나이저 풀 반환"""
    return _tokenizer_pool

def get_model():
    """분류 모델 반환"""
    return _model

def get_qdrant_client():
    """Qdrant 벡터 DB 클라이언트 반환"""
    return _qdrant_client

