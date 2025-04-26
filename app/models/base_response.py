# models/base_response.py

from pydantic import BaseModel
from typing import Generic, TypeVar, List, Optional
from datetime import datetime, timezone

T = TypeVar("T")

class BaseResponseModel(BaseModel, Generic[T]):
    results: List[T]
    count: int
    success: bool = True
    timestamp: datetime = datetime.now(timezone.utc)
