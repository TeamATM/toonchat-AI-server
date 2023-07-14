from app.models import ORJSONMoel
from pydantic import UUID4


class RequestData(ORJSONMoel):
    session_id: str
    input: str


class RedisKeyResponse(ORJSONMoel):
    redis_key: UUID4


class CompletionResponse(ORJSONMoel):
    result: str
