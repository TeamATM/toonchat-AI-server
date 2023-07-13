from app.models import ORJSONMoel
from pydantic import UUID4


class RequestData(ORJSONMoel):
    session_id: str
    input: str


class InferenceResponse(ORJSONMoel):
    redis_key: UUID4
